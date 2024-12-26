import torchvision
import torchvision.transforms as TF
import torch
import os
import shutil
import numpy as np
from torch.utils.data import DataLoader
from VAE import TrainingConfig, ModelConfig, VAETrainer, VAE


if __name__ == "__main__":
    def seed_everything(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
            torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积结果固定
            torch.backends.cudnn.benchmark = False

    # 使用
    seed_everything(46)
    transform_vae = TF.Compose(
    [TF.ToTensor(),
        TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_vae = TF.Compose(
    # [TF.ToTensor()])
# 数据范围[-1,1]
    trainset_vae = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=transform_vae)
    testset_vae = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                            download=True, transform=transform_vae)
    # 配置参数
    trainconf = TrainingConfig()
    modelconf = ModelConfig()
    
    # 准备数据
    vae_trainloader = DataLoader(trainset_vae, batch_size=trainconf.BATCH_SIZE, shuffle=True)
    vae_testloader = DataLoader(testset_vae, batch_size=trainconf.BATCH_SIZE, shuffle=True)
    
    # 初始化模型和优化器
    vae_model = VAE(modelconf.LATENT_CHANNEL, modelconf.DOWNSAMPLE).to(trainconf.DEVICE)
    vae_params = []
    for name, module in vae_model.named_children():
        if name != 'discriminator':  # 排除判别器的参数
            vae_params.extend(list(module.parameters()))
    vae_optimizer  = torch.optim.Adam(params=vae_params, lr=trainconf.VAE_LR, betas=(trainconf.BETA1, trainconf.BETA2), eps=trainconf.EPS)
    disc_optimizer = torch.optim.Adam(params=vae_model.discriminator.parameters(), lr=trainconf.DISC_LR, betas=(trainconf.BETA1, trainconf.BETA2), eps=trainconf.EPS )

    # 初始化训练器
    trainer = VAETrainer(vae_model, trainconf)


     # 设置目录
    save_dirs = trainer.setup_directories()

    train_psnr_history = []
    test_psnr_history = []
    
    # 训练循环
    for epoch in range(trainconf.NUM_EPOCHS):
        epoch_train_psnr = trainer.train_epoch(vae_trainloader, vae_optimizer, disc_optimizer, 
                          epoch, trainconf.NUM_EPOCHS, save_dirs)
        
        epoch_test_psnr = trainer.test_epoch(vae_testloader, epoch, trainconf.NUM_EPOCHS, save_dirs)
        
        # 记录PSNR历史
        train_psnr_history.append(epoch_train_psnr)
        test_psnr_history.append(epoch_test_psnr)
        
        # 打印当前epoch的PSNR
        print(f"\nEpoch {epoch + 1} Train PSNR metrics:")
        print(f"Train PSNR: {epoch_train_psnr:.2f}")
        print(f"\nEpoch {epoch + 1} Test PSNR metrics:")
        print(f"Test PSNR: {epoch_test_psnr:.2f}")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = trainer._save_checkpoint(epoch, save_dirs['checkpoints'], 
                                                    psnr_history=train_psnr_history)
    
    # 保存最终模型
    final_model_path = os.path.join(save_dirs['model'], "final_vae_model.pt")
    shutil.copy2(checkpoint_path, final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")