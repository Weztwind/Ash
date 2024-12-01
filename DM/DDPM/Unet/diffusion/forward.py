import torch
from .diffusion import SimpleDiffusion
from ..tools.helper import get



def forward_diffusion(sd: SimpleDiffusion, x0: torch.Tensor, timesteps: torch.Tensor):
    """
    前向扩散过程的函数。

    Args:
        sd: SimpleDiffusion 对象,表示简单的扩散过程。
        x0: 原始图像数据。
        timesteps: 一个张量,表示扩散过程中的时间步。

    Returns:
        sample: 扩散后的图像。
        eps: 高斯噪声。
    """
    eps = torch.randn_like(x0)  # 生成与原始图像形状相同的高斯噪声
    mean = get(sd.sqrt_alpha_cumulative, t=timesteps) * x0  # 根据时间步计算扩散过程中的均值
    std_dev = get(sd.sqrt_one_minus_alpha_cumulative, t=timesteps)  # 根据时间步计算扩散过程中的标准差
    sample = mean + std_dev * eps  # 将均值和标准差按比例添加到原始图像上,得到扩散后的图像

    return sample, eps # 返回扩散后的图像和高斯噪声以及经过嵌入后的标签