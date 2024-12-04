import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, channels=64, label_dim=128):
        super().__init__()
        self.channels = channels
        self.label_dim = label_dim

        self.group_norm_x = nn.GroupNorm(num_groups=8, num_channels=channels)  # 对输入 x 进行的组归一化
        self.group_norm_y = nn.GroupNorm(num_groups=8, num_channels=label_dim)  # 对输入 y 进行的组归一化
        self.label_proj = nn.Linear(label_dim, channels)  # 将 label_dim 投射到 channels 维度
        self.mhsa = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)  # 多头注意力机制

    def forward(self, x, y=None):
        B, _, H, W = x.shape

        # 对 x 进行批归一化并调整形状
        h_x = self.group_norm_x(x)
        h_x = h_x.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] -> [B, H * W, C]

        if y is None:
            # 自注意力
            h, _ = self.mhsa(h_x, h_x, h_x)  # Query, Key, Value 都是 h_x
        else:
            # 检查 y 的形状，如果是二维的，则调整形状
            if len(y.shape) == 2:
                y = y.unsqueeze(2).unsqueeze(3)  # [B, label_dim] -> [B, label_dim, 1, 1]
            elif len(y.shape) != 4:
                raise ValueError(f'Expected y to have 2 or 4 dimensions, but got {len(y.shape)}')

            # 对 y 进行批归一化并调整形状
            h_y = self.group_norm_y(y)
            h_y = self.label_proj(h_y.reshape(B, self.label_dim))  # [B, label_dim, 1, 1] -> [B, channels]
            h_y = h_y.unsqueeze(1)  # [B, channels] -> [B, 1, channels]

            # 交叉注意力
            h, _ = self.mhsa(h_x, h_y, h_y)  # Query 是 h_x，Key 和 Value 是 h_y

        # 将注意力输出 h 调整回原始形状
        h = h.swapaxes(1, 2).reshape(B, self.channels, H, W)  # [B, H * W, C] -> [B, C, H, W]

        # 返回 x + h，实现残差连接
        return x + h
