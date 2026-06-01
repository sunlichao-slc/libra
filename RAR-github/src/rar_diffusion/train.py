"""Training loop for conditional DDPM on VQ-VAE latents."""

from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, VQModel

from rar_diffusion.checkpoints import save_denoiser
from rar_diffusion.config import TrainingConfig
from rar_diffusion.embeddings import ensure_3d_embedding


def get_last_saved_epoch(output_dir: str | os.PathLike) -> int:
    output_dir = os.fspath(output_dir)
    if not os.path.isdir(output_dir):
        return -1
    epoch_dirs = [d for d in os.listdir(output_dir) if re.match(r"epoch-\d+", d)]
    if not epoch_dirs:
        return -1
    return max(int(re.search(r"epoch-(\d+)", d).group(1)) for d in epoch_dirs)


def load_model_from_last_epoch(
    output_dir: str | os.PathLike, model: UNet2DConditionModel, vqvae: VQModel
):
    last_epoch = get_last_saved_epoch(output_dir)
    if last_epoch >= 0:
        epoch_dir = os.path.join(output_dir, f"epoch-{last_epoch}")
        model = UNet2DConditionModel.from_pretrained(epoch_dir)
        vqvae = VQModel.from_pretrained(epoch_dir)
    return model, vqvae, last_epoch


def train_loop(
    config: TrainingConfig,
    model,
    vqvae,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    eval_embedding: torch.Tensor,
    test_embedding: torch.Tensor,
):
    accelerator = Accelerator(
        device_placement=True,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    device = accelerator.device
    eval_embedding = ensure_3d_embedding(eval_embedding).to(device)
    test_embedding = ensure_3d_embedding(test_embedding).to(device)

    if accelerator.is_main_process and config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.eval_images_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, vqvae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, vqvae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    start_epoch = 0

    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for batch in train_dataloader:
            images, embeddings = batch
            vqvae_module = getattr(vqvae, "module", vqvae)
            with torch.no_grad():
                latents = vqvae_module.encode(images).latents

            noise = torch.randn(latents.shape, device=latents.device)
            bs = images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=images.device,
                dtype=torch.int64,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            cond = ensure_3d_embedding(embeddings).to(noisy_latents.dtype)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_latents, timesteps, cond, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if epoch % config.save_image_epochs == 0 and accelerator.is_main_process:
            _sample_and_save(
                model,
                vqvae,
                noise_scheduler,
                latents,
                eval_embedding,
                test_embedding,
                config.eval_images_dir,
                epoch,
                accelerator,
            )

        if epoch % config.save_model_epochs == 0 and accelerator.is_main_process:
            epoch_output_dir = os.path.join(config.output_dir, f"epoch-{epoch}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            save_denoiser(accelerator.unwrap_model(model), epoch_output_dir)

    if accelerator.is_main_process:
        print("Training completed.")


def _sample_and_save(
    model,
    vqvae,
    noise_scheduler,
    latents,
    eval_embedding,
    test_embedding,
    eval_dir: str,
    epoch: int,
    accelerator: Accelerator,
):
    os.makedirs(eval_dir, exist_ok=True)
    vqvae_unwrapped = accelerator.unwrap_model(vqvae)

    with torch.no_grad():
        for name, embedding in [("eval", eval_embedding), ("eval-test", test_embedding)]:
            sample_latents = torch.randn(1, *latents.shape[1:], device=latents.device)
            for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                model_timestep = torch.tensor([t], device=latents.device)
                noise_pred = model(sample_latents, model_timestep, embedding, return_dict=False)[0]
                sample_latents = noise_scheduler.step(
                    noise_pred, t, sample_latents, return_dict=True
                ).prev_sample

            image = vqvae_unwrapped.decode(sample_latents).sample.squeeze(0).cpu().permute(1, 2, 0)
            image = (image * 127.5 + 127.5).clamp(0, 255).numpy().astype("uint8")
            plt.imsave(os.path.join(eval_dir, f"{name}-{epoch}.jpg"), image)

    torch.cuda.empty_cache()
