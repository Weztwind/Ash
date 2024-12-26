import torch.nn as nn
import torch
from .LPIPS_layer import NetLinLayer, ScalingLayer, VGG16
from ..tools import get_ckpt_path, norm_tensor, spatial_average

class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.vgg = VGG16()
        self.lins = nn.ModuleList([
            NetLinLayer(self.channels[0]),
            NetLinLayer(self.channels[1]),
            NetLinLayer(self.channels[2]),
            NetLinLayer(self.channels[3]),
            NetLinLayer(self.channels[4])
        ])

        self.load_from_pretrained()

        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "vgg_lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)

    # def forward(self, real_x, fake_x):
    #     features_real = self.vgg(self.scaling_layer(real_x))
    #     features_fake = self.vgg(self.scaling_layer(fake_x))
    #     diffs = {}

    #     for i in range(len(self.channels)):
    #         diffs[i] = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2

    #     s = sum([spatial_average(self.lins[i].model(diffs[i])) for i in range(len(self.channels))])


    #     return s

    def forward(self, real_x, fake_x):
        features_real = self.vgg(self.scaling_layer(real_x))
        features_fake = self.vgg(self.scaling_layer(fake_x))
        
        layer_distances = []
        for i in range(len(self.channels)):
            diff = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2
            weighted = self.lins[i].model(diff)
            avg = spatial_average(weighted)
            layer_distances.append(torch.abs(avg))  # 每层的结果取绝对值

        s = sum(layer_distances)  # 和一定为正
        return s