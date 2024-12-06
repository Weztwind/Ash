import torchvision
import torchvision.transforms as TF
import torch
import os
import shutil
from torch.utils.data import DataLoader
# from LDM_VAE import TrainingConfig, VAE, ModelConfig, VAETrainer
from VAE import TrainingConfig, ModelConfig, VAETrainer, VAE


if __name__ == "__main__":
    transform_vae = TF.Compose(
    [TF.ToTensor(),
        TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 数据范围[-1,1]
    trainset_vae = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=transform_vae)
    testset_vae = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                            download=True, transform=transform_vae)
    # 配置参数
    trainconf = TrainingConfig()
    modelconf = ModelConfig()
    
    # 准备数据
    vae_dataloader = DataLoader(trainset_vae, batch_size=trainconf.BATCH_SIZE, shuffle=True)
    
    # 初始化模型和优化器
    vae_model = VAE(modelconf.LATENT_CHANNEL, modelconf.DOWNSAMPLE).to(trainconf.DEVICE)
    vae_params = []
    for name, module in vae_model.named_children():
        if name != 'discriminator':  # 排除判别器的参数
            vae_params.extend(list(module.parameters()))
    vae_optimizer = vae_optimizer = torch.optim.Adam(params=vae_params, lr=trainconf.LR, betas=(trainconf.BETA1, trainconf.BETA2), eps=trainconf.EPS)
    disc_optimizer = torch.optim.Adam(params=vae_model.discriminator.parameters(), lr=trainconf.LR, eps=trainconf.EPS, betas=(trainconf.BETA1, trainconf.BETA2))

    # 初始化训练器
    trainer = VAETrainer(vae_model, trainconf)


     # 设置目录
    save_dirs = trainer.setup_directories()
    
    # 训练循环
    for epoch in range(trainconf.NUM_EPOCHS):
        trainer.train_epoch(vae_dataloader, vae_optimizer, disc_optimizer, 
                          epoch, trainconf.NUM_EPOCHS, save_dirs)
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = trainer._save_checkpoint(epoch, save_dirs['checkpoints'])
    
    # 保存最终模型
    final_model_path = os.path.join(save_dirs['model'], "final_vae_model.pt")
    shutil.copy2(checkpoint_path, final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")