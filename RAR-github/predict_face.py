import os
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DConditionModel, VQModel, DDPMScheduler
import matplotlib.pyplot as plt

# Paths
embeddings_dir = "/home/wuyuncheng/RAR/embedding_resnet/cub200/001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.npy"  # 嵌入文件目录
output_dir = "/home/wuyuncheng/RAR/reverse-face-latent-diffusion-main/test_data/"  # 输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载训练好的模型
# unet_path = "/home/wuyuncheng/RAR/models/128/small-unet"  # 替换为你的UNet模型路径
# vqvae = VQModel.from_pretrained("models/128/small-vqvae")  # 加载预训练的VQ-VAE
# unet = UNet2DConditionModel.from_pretrained(unet_path)  # 加载UNet模型
unet = UNet2DConditionModel.from_pretrained("/home/wuyuncheng/RAR/reverse-face-latent-diffusion-main/train_model/epoch-495/epoch-500_cub200_resnet")
# vqvae = VQModel.from_pretrained("/home/wuyuncheng/RAR/RAR-main/config/ldm-super", subfolder="vqvae")
vqvae = VQModel.from_pretrained("/home/wuyuncheng/RAR/reverse-face-latent-diffusion-main/test_data/caletch_128/small-vqvae")  # 加载预训练的VQ-VAE

# 定义扩散调度器
scheduler = DDPMScheduler(
    beta_start=0.0015,
    beta_end=0.0195,
    beta_schedule="scaled_linear",
    clip_sample=False,
    prediction_type="epsilon",
    num_train_timesteps=1000
)

# 将模型移动到设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae.to(device)
unet.to(device)
scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

# 将模型设置为评估模式
vqvae.eval()
unet.eval()

# 定义函数：从潜在变量解码为图像
def decode_latents(latents):
    with torch.no_grad():
        images = vqvae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)  # 将图像缩放到 [0, 1]
    return images

# 定义函数：从单个嵌入生成图像
def inference_single(embedding):
    with torch.no_grad():
        latent_size = 16  # 基于VQ-VAE的潜在分辨率

        # 初始化随机潜在变量
        holdout_latents = torch.randn(
            (1, vqvae.config.latent_channels, latent_size, latent_size),
            device=device,
        )

        # 反向扩散过程
        for t in reversed(range(scheduler.config.num_train_timesteps)):
            timesteps = torch.tensor([t], device=device)
            noise_pred = unet(holdout_latents, timesteps, embedding, return_dict=False)[0]
            holdout_latents = scheduler.step(noise_pred, timesteps, holdout_latents).prev_sample

        # 将潜在变量解码为图像
        pred_image = vqvae.decode(holdout_latents).sample  # [1, C, H, W]
        pred_image = pred_image.permute(0, 2, 3, 1).cpu().numpy()  # 转换为 [1, H, W, C]
        pred_image = (pred_image * 127.5 + 127.5).clip(0, 255).astype("uint8")  # 缩放到 [0, 255]
        return pred_image[0]  # 返回单张图像

# 定义函数：从单个嵌入文件生成图像并保存
def predict_single_image(embedding_file):
    # 加载嵌入文件
    embedding_path = os.path.join(embeddings_dir, embedding_file)
    embedding = np.load(embedding_path)
    embedding = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # 添加维度

    # 生成图像
    pred_image = inference_single(embedding)  # [H, W, C]

    # 保存图像
    output_path = os.path.join(output_dir, f"{embedding_file.replace('.npy', '')}.jpg")
    plt.imsave(output_path, pred_image)
    print(f"图像已保存到: {output_path}")

# 示例：从单个嵌入文件生成图像
embedding_file = "//home/wuyuncheng/RAR/RAR-main/panda_image_embedding.npy"  # 替换为你的嵌入文件名
predict_single_image(embedding_file)
