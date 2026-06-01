from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size: int = 256
    train_batch_size: int = 32
    eval_batch_size: int = 4
    num_epochs: int = 1001
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    save_model_epochs: int = 5
    mixed_precision: str = "fp16"
    output_dir: str = "outputs/checkpoints/ddpm-retrain"
    overwrite_output_dir: bool = True
    seed: int = 0
    eval_images_dir: str = "outputs/eval_images"
