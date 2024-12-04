import torch
import torch.nn as nn
from .embeddings import SinusoidalPositionEmbeddings, LabelEmbeddings
from .blocks import ResnetBlock, DownSample, UpSample



class UNet(nn.Module):
    def __init__(
        self,
        input_channels=3,  # 输入图像的通道数,默认为3(RGB)
        output_channels=3,  # 输出图像的通道数,默认为3(RGB)
        num_res_blocks=2,  # 每个级别的ResNet块的数量，Deocder层最终使用的UpBlock数=num_res_blocks + 1
        base_channels=128,  # 第一层卷积的输出通道数
        base_channels_multiples=(1, 2, 4, 8),  # 每个层的通道数相对于第一层卷积输出通道数的倍数
        apply_attention=(False, False, True, False),  # 在每个层是否应用注意力机制
        dropout_rate=0.1,  # ResNet块中使用的dropout率
        time_multiple=4,  # 时间嵌入维度相对于base_channels的倍数
        label_multiple=4,  # 标签嵌入维度相对于base_channels的倍数
        num_classes=10,


    ):
        super().__init__()

        # 计算时间嵌入的维度
        time_emb_dims_exp = base_channels * time_multiple
        # 创建正弦位置嵌入层
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)

        # 计算标签嵌入的维度
        label_emb_dims_exp = base_channels * label_multiple
        ## 创建标签嵌入层
        self.label_embeddings = LabelEmbeddings(num_classes=num_classes, embed_dim=label_emb_dims_exp)

        # 第一个卷积层,将输入通道数转换为基础通道数
        self.first = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")

        num_resolutions = len(base_channels_multiples)#分辨率数量，因为每次下采样或者上采样都会改变分辨率，也就是encoder和decoder的层数

        # UNet的编码器部分,进行维度缩减
        self.encoder_blocks = nn.ModuleList()  # 创建一个空的ModuleList,用于存储编码器部分的所有层
        curr_channels = [base_channels]  # 初始化一个列表,用于跟踪当前级别的通道数,初始值为base_channels,方便后续decoder确定通道数
        in_channels = base_channels  # 初始化当前级别的输入通道数为base_channels

        for level in range(num_resolutions):  # 对于每一层
            out_channels = base_channels * base_channels_multiples[level]  # 计算当前级别的输出通道数

            for _ in range(num_res_blocks):  # 对于每个ResNet块
                # 创建ResNet块
                block = ResnetBlock(
                    in_channels=in_channels,  # 当前ResNet块的输入通道数
                    out_channels=out_channels,  # 当前ResNet块的输出通道数
                    dropout_rate=dropout_rate,  # ResNet块中使用的dropout率
                    time_emb_dims=time_emb_dims_exp,  # 时间嵌入的维度
                    label_emb_dims=label_emb_dims_exp,  # 标签嵌入的维度
                    apply_attention=apply_attention[level],  # 是否在当前级别应用注意力机制
                )
                self.encoder_blocks.append(block)  # 将创建的ResNet块添加到编码器的ModuleList中

                in_channels = out_channels  # 更新下一个ResNet块的输入通道数为当前ResNet块的输出通道数
                curr_channels.append(in_channels)  # 将当前级别的输出通道数添加到curr_channels列表中,用于跟踪编码器每个级别的输出通道数

            if level != (num_resolutions - 1):
                # 在编码器的每个级别(除了最后一级)添加下采样层
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # 中间的瓶颈块
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    label_emb_dims=label_emb_dims_exp,  # 标签嵌入的维度
                    apply_attention=True,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    label_emb_dims=label_emb_dims_exp,  # 标签嵌入的维度
                    apply_attention=False,
                ),
            )
        )

        # UNet的解码器部分,进行维度恢复,并使用跳跃连接
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks + 1):#解码器每一层的resnet块数等于对应编码器层的块数+1
                encoder_in_channels = curr_channels.pop()
                # 创建ResNet块,并将编码器的输出与解码器的输入连接
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,#跳跃连接
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    label_emb_dims=label_emb_dims_exp,  # 标签嵌入的维度
                    apply_attention=apply_attention[level],
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                # 在解码器的每个级别(除了最后一级)添加上采样层
                self.decoder_blocks.append(UpSample(in_channels))

        # 最后的输出块,将通道数转换为输出通道数
        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, x, y, t):
        # 获取时间步长的嵌入
        time_emb = self.time_embeddings(t)
        # 获取标签的嵌入
        label_emb = self.label_embeddings(y)


        # 第一个卷积层
        h = self.first(x)
        outs = [h]

        # 编码器部分
        for layer in self.encoder_blocks:
            if isinstance(layer, ResnetBlock):
                h = layer(h, time_emb, label_emb)
            else:
                h = layer(h, time_emb)
            outs.append(h)

        # 瓶颈块
        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb, label_emb)

        # 解码器部分
        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):#判断是否是resnetblock，跳跃连接，而上采用则不需要
                out = outs.pop()
                h = torch.cat([h, out], dim=1)  # 连接编码器的输出和解码器的输入
                h = layer(h, time_emb, label_emb)
            else:
                h = layer(h, time_emb)



        # 最后的输出块
        h = self.final(h)

        return h
