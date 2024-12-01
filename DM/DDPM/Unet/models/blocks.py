import torch
import torch.nn as nn
from .attention import AttentionBlock

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512,label_emb_dims=512, apply_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_fn = nn.SiLU()  # 使用 SiLU 激活函数

        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)  # 批归一化层
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=3, stride=1, padding="same")  # 卷积层

        # Group 2 time embedding
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)  # 全连接层

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)  # 批归一化层
        self.dropout = nn.Dropout2d(p=dropout_rate)  # 二维随机丢弃层
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                                kernel_size=3, stride=1, padding="same")  # 卷积层

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=1, stride=1)  # 卷积层用于匹配输入和输出通道数
        else:
            self.match_input = nn.Identity()  # 恒等映射，用于通道数已匹配的情况

        if apply_attention:
            # self.attention = AttentionBlock(channels=self.out_channels)  # 注意力块
            self.attention = AttentionBlock(channels=self.out_channels,label_dim=label_emb_dims) # 交叉注意力块
        else:
            self.attention = nn.Identity()  # 不应用注意力机制，恒等映射

    def forward(self, x, t, y):
        ##y代表标签
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        # group 2
        # add in timestep embedding
        h += self.dense_1(self.act_fn(t))[:, :, None, None]

        # group 3
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention
        h = h + self.match_input(x)
        if isinstance(self.attention, AttentionBlock):
            h = self.attention(h, y)
        else:
            h = self.attention(h)

        return h
    
class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)