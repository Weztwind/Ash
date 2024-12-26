# Diffusion Model Learning

该仓库整理了现有的DM相关算法，供个人学习使用。

## 更新日志
### 2024/11/29
- 创建仓库
- DM\DDPM\Unet目录结构梳理
- 更新了DM\DDPM\Unet\models
### 2024/12/01
- 更新了DM\DDPM\Unet\diffusion
- 更新了DM\DDPM\Unet\tools
- 更新了DM\DDPM\Unet\training
### 2024/12/04
- 更新了目录结构，基于UNET的DDPM可通过DM\DDPM\ddpm_unet.py运行
### 2024/12/06
- 更新了LDM_VAE, 存在问题：VAE训练效果差
### 2024/12/26
- 更新了LDM_VAE, 添加了不同loss函数：L1/L2 loss, perceptual loss, gan loss, kl loss。训练时需要把kl loss的权重设置得非常小 
