import torch
import torchvision.transforms as TF
from PIL import Image
from ..tools import get, make_grid, inverse_transform, frames2vid
from IPython.display import display
from tqdm import tqdm
from ..training import TrainingConfig, BaseConfig

@torch.inference_mode()
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64),
                      num_images=10, nrow=10, device=BaseConfig.DEVICE, num_classes=TrainingConfig.NUM_CLASS , is_latent = False, vae_model = None, **kwargs):
    """
    反向扩散过程的函数,用于从随机噪声生成图像。

    Args:
        model: 训练好的条件扩散模型。
        sd: SimpleDiffusion 对象,表示简单的扩散过程。
        timesteps: 反向扩散过程的总时间步数。
        img_shape: 生成图像的形状。
        num_images: 每个类别要生成的图像数量。
        nrow: 在生成的网格图像中每行显示的图像数量。
        device: 使用的设备(CPU 或 GPU)。
        **kwargs: 其他可选参数,如 generate_video 和 save_path。

    Returns:
        None
    """
    if is_latent and vae_model is None:
        raise ValueError("VAE model is required when is_latent=True")
    
    total_images = num_images * num_classes  # 总图像数量
    x = torch.randn((total_images, *img_shape), device=device)  # 初始化随机噪声

    # 创建类别标签，每个类别生成 num_images 个图像
    y = torch.arange(num_classes, device=device).repeat_interleave(num_images)

    model.eval()  # 将模型设置为评估模式
    if is_latent:
        vae_model.eval()
    

    if kwargs.get("generate_video", False):
        outs = []  # 用于存储生成的图像帧

    # 反向迭代时间步
    for time_step in tqdm(iterable=reversed(range(1, timesteps)), total=timesteps - 1, dynamic_ncols=False, desc="Sampling :: ", position=0):
        ts = torch.ones(total_images, dtype=torch.long, device=device) * time_step  # 创建当前时间步的张量
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)  # 创建随机噪声或零噪声

        predicted_noise = model(x, y, ts)  # 使用模型预测噪声

        # 计算反向扩散过程中的系数
        beta_t = get(sd.beta, ts)
        one_by_sqrt_alpha_t = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts)

        # 根据预测的噪声和系数更新图像
        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )




        if kwargs.get("generate_video", False):
            if is_latent:
                x_decoded = vae_model.decode(x)
                x_inv = inverse_transform(x_decoded).type(torch.uint8)  # 对生成的图像进行反向变换
            else:
                x_inv = inverse_transform(x).type(torch.uint8)
            # x_inv = inverse_transform(x_decoded).type(torch.uint8)  # 对生成的图像进行反向变换
            grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")  # 创建图像网格
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]  # 调整图像的维度顺序
            outs.append(ndarr)  # 将图像帧添加到列表中

    if kwargs.get("generate_video", False):  # 如果需要生成视频
        frames2vid(outs, kwargs['save_path'])  # 将图像帧转换为视频并保存
        display(Image.fromarray(outs[-1][:, :, ::-1]))  # 显示反向扩散过程最后一步的图像
        return None

    else:  # 如果不生成视频,只保存和显示最后一步的图像
        if is_latent:
            x_decoded = vae_model.decode(x)
            x_inv = inverse_transform(x_decoded).type(torch.uint8)  # 对生成的图像进行反向变换

        else:
            x_inv = inverse_transform(x).type(torch.uint8)
        # x = inverse_transform(x_decoded).type(torch.uint8)  # 对生成的图像进行反向变换
        grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")  # 创建图像网格
        pil_image = TF.functional.to_pil_image(grid)  # 将图像转换为 PIL 图像
        save_path = kwargs['save_path']
        pil_image.save(kwargs['save_path'], format=save_path[-3:].upper())  # 保存图像
        display(pil_image)  # 显示图像
        return None