import torch.nn as nn
import torch
import torch.nn.functional as F
from .LPIPS import LPIPS
from .Discrim import Discriminator


class VAE(nn.Module):
    def __init__(self, latent_c, downsample=1):
        super(VAE, self).__init__()

        self.latent_c = latent_c
        self.downsample = downsample

        self.perceptual_loss = LPIPS().eval()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # First downsampling by factor of 2
        )

        # 根据 downsample 增加额外的池化层
        for _ in range(self.downsample - 1):
            self.encoder.add_module("extra_pool_{}".format(_), nn.MaxPool2d(kernel_size=2, stride=2))

        # 均值和标准差卷积层
        self.fc_mu = nn.Conv2d(32, latent_c, kernel_size=1)
        self.fc_logvar = nn.Conv2d(32, latent_c, kernel_size=1)
        # # 解码器
        self.decoder = nn.Sequential()
        # 根据 downsample 添加相应数量的上采样+卷积层
        for _ in range(self.downsample - 1):
            # 添加双线性上采样模块
            self.decoder.add_module("upsample_{}".format(_),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            # 添加卷积层以及ReLU激活层
            self.decoder.add_module("conv_{}".format(_),
                                    nn.Conv2d(latent_c, latent_c, kernel_size=3, stride=1, padding=1))
            self.decoder.add_module("leakyrelu_{}".format(_), nn.LeakyReLU(0.2))
            # self.decoder.add_module("relu_{}".format(_), nn.ReLU())

        # 最后的上采样和卷积层
        self.decoder.add_module("final_upsample", nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.decoder.add_module("final_conv", nn.Conv2d(latent_c, 32, kernel_size=3, stride=1, padding=1))
        self.decoder.add_module("final_leakyrelu", nn.LeakyReLU(0.2))
        # self.decoder.add_module("final_relu", nn.ReLU())

        # 输出卷积层
        self.decoder.add_module("output_conv", nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1))
        # self.decoder.add_module("output_sigmoid", nn.Sigmoid())
        self.decoder.add_module("output_Tanh", nn.Tanh())

        # 判别器
        self.discriminator = Discriminator(image_channels=3)
    
    

    def to(self, device):
        super().to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        return self


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # 通过编码器
        encoded = self.encoder(x)

        # 提取均值和对数方差
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # 采样潜在向量
        z = self.reparameterize(mu, logvar)

        # 通过解码器还原图像
        reconstructed = self.decoder(z)

        # 调整输出为输入的尺寸
        reconstructed = F.interpolate(reconstructed, size=(x.size(2), x.size(3)), mode='nearest')

        return reconstructed, mu, logvar

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def discriminator_forward(self, x):
        # 通过判别器进行前向传播
        return self.discriminator(x)

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def calculate_lambda(self, perceptual_loss, gan_loss):
        # 在这里，我们假设解码器的最后一个卷积层是输出层前的卷积层，即 'output_conv'
        last_layer = self.decoder.output_conv

        # 获取最后一层的权重
        last_layer_weight = last_layer.weight

        # 计算感知损失与GAN损失关于这个权重的梯度
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        # 计算λ（lambda）的值
        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)  # 添加小数避免除零
        λ = torch.clamp(λ, 0, 1e4).detach()  # 限制λ的范围并从计算图中分离

        # 返回调整后的λ值
        return 0.8 * λ