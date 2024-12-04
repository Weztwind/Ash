import torch
import glob
import torch.nn as nn
import os
import torchvision
import torchvision.transforms as TF
from DDPM_Unet import UNet, TrainingConfig, ModelConfig, BaseConfig, train_one_epoch, SimpleDiffusion, reverse_diffusion, load_checkpoint
from torch.cuda import amp
from torch.utils.data import DataLoader








def scale_to_range(t):
    return (t * 2) - 1

if __name__ == '__main__':
    # 定义图像预处理操作

    

    transform_dm = TF.Compose(
    [
        TF.ToTensor(),
        TF.Resize((32, 32),
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True),
        TF.RandomHorizontalFlip(),
        TF.Lambda(scale_to_range)
    ]
)
    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_dm)
    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_dm)
    dm_dataloader = DataLoader(trainset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)

    source_path = "./results"
    dm_save_path = os.path.join(source_path, f"dm")
    os.makedirs(dm_save_path, exist_ok=True)

    is_dm_train = True
    is_dm_inference = True
    dm_model = UNet(  ##网络采用unet结构
                input_channels=TrainingConfig.IMG_SHAPE[0],  # 输入图像的通道数
                output_channels=TrainingConfig.IMG_SHAPE[0],  # 输出图像的通道数
                base_channels=ModelConfig.BASE_CH,
                base_channels_multiples=ModelConfig.BASE_CH_MULT,
                apply_attention=ModelConfig.APPLY_ATTENTION,
                dropout_rate=ModelConfig.DROPOUT_RATE,
                time_multiple=ModelConfig.TIME_EMB_MULT,
            )
    dm_model.to(BaseConfig.DEVICE)

    dm_optimizer = torch.optim.AdamW(dm_model.parameters(), lr=TrainingConfig.LR)

    loss_fn = nn.MSELoss()
    sd = SimpleDiffusion(
            num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
            img_shape=TrainingConfig.IMG_SHAPE,
            device=BaseConfig.DEVICE,
        )

    scaler = amp.GradScaler()

    total_epochs = TrainingConfig.NUM_EPOCHS
    if is_dm_train:
        # 训练开始前，尝试加载最新的checkpoint
        latest_checkpoint = max(glob.glob(os.path.join(dm_save_path, "checkpoints/checkpoint_epoch_*.pt")), default=None)
        if latest_checkpoint:
            start_epoch = load_checkpoint(latest_checkpoint, dm_model, dm_optimizer, scaler)
        else:
            start_epoch = 1
        # 算法 1: 训练
        for epoch in range(1, total_epochs + 1):
            train_one_epoch(dm_model, sd, dm_dataloader, dm_optimizer, scaler, loss_fn, epoch, total_epochs)  # 调用 train_one_epoch 函数进行一个 epoch 的训练

            # 每N轮保存一次checkpoint
            if epoch % TrainingConfig.CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': dm_model.state_dict(),
                    'optimizer_state_dict': dm_optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'loss': loss_fn
                }
                checkpoint_path = os.path.join(dm_save_path, f"checkpoints/checkpoint_epoch_{epoch}.pt")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch}")

                # 只保留最近的N个checkpoint，删除旧的
                checkpoints = sorted(glob.glob(os.path.join(dm_save_path, "checkpoints/checkpoint_epoch_*.pt")))
                if len(checkpoints) > TrainingConfig.MAX_CHECKPOINTS:
                    os.remove(checkpoints[0])

            
            if (epoch % 20 == 0) or (epoch == total_epochs) or (epoch == 1):  
                log_dir = os.path.join(dm_save_path, f"Inference" )
                os.makedirs(log_dir, exist_ok = True)
                dm_pic_save_path = os.path.join(log_dir, f"{epoch}{TrainingConfig.EXT}")  # 构建保存生成图像的路径

                # 算法 2: 采样
                reverse_diffusion(dm_model, 
                                sd, 
                                timesteps = TrainingConfig.TIMESTEPS, 
                                img_shape = TrainingConfig.IMG_SHAPE, 
                                num_images = 5,
                                device = BaseConfig.DEVICE, 
                                num_classes = TrainingConfig.NUM_CLASS, 
                                is_latent = False, 
                                generate_video = TrainingConfig.GENERATE_VIDEO, 
                                save_path = dm_pic_save_path
                                )  # 调用 reverse_diffusion 函数进行采样,生成图像
        
        
        final_model_save_path = os.path.join(dm_save_path, f"final_models/final_model.pt")

        # 确保保存路径存在
        os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True)

        # 保存模型
        torch.save(dm_model.state_dict(), final_model_save_path)
        print(f"Saved final model at {final_model_save_path}")