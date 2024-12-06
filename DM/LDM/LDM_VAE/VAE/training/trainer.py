import torch
import os
import torch.nn.functional as F
from torchvision.utils import save_image


from tqdm import tqdm

class VAETrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def _compute_losses(self, data, recon_batch, mu, logvar, disc_real, disc_fake, disc_factor):
        """计算所有损失函数"""
        loss_perceptual = self.model.perceptual_loss(data, recon_batch)
        loss_recon = torch.abs(data - recon_batch)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_per_recon_kl = (loss_perceptual + loss_recon + 0.0001*loss_kl).mean()
        loss_g = -torch.mean(disc_fake)
        
        λ = self.model.calculate_lambda(loss_per_recon_kl, loss_g)
        loss_vae = loss_per_recon_kl + self.config.DISCFACTOR * λ * loss_g
        
        loss_d_real = torch.mean(F.relu(1. - disc_real))
        loss_d_fake = torch.mean(F.relu(1. + disc_fake))
        loss_gan = disc_factor * 0.5 * (loss_d_real + loss_d_fake)
        
        return loss_vae, loss_gan
    
    def optimization_step(self, vae_optimizer, disc_optimizer, loss_vae, loss_gan):
        vae_optimizer.zero_grad()
        loss_vae.backward(retain_graph=True)
        disc_optimizer.zero_grad()
        loss_gan.backward()
        vae_optimizer.step()
        disc_optimizer.step()
        
    def _save_samples(self, data, recon_batch, epoch, batch_idx, save_dir):
        """保存重构样本"""
        with torch.no_grad():
            comparison = torch.cat((data[:4], recon_batch[:4]))
            save_image((comparison + 1) / 2, 
                      os.path.join(save_dir, f"recon_epoch{epoch+1}_batch{batch_idx}.png"), 
                      nrow=4)
    
    def _save_checkpoint(self, epoch, save_dir):
        """保存模型检查点"""
        checkpoint_path = os.path.join(save_dir, f"vae_epoch_{epoch+1}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path
        
    def train_epoch(self, data_loader, vae_optimizer, disc_optimizer, epoch, total_epochs, save_dirs):
        """训练一个epoch"""
        steps_per_epoch = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=steps_per_epoch, 
                   desc=f"Epoch {epoch + 1}/{total_epochs}")
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.config.DEVICE, dtype=torch.float)
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            disc_real = self.model.discriminator(data)
            disc_fake = self.model.discriminator(recon_batch)
            
            # 计算判别器因子
            disc_factor = self.model.adopt_weight(self.config.DISCFACTOR, 
                                                epoch * steps_per_epoch + batch_idx,
                                                threshold=self.config.DISC_START)
            
            # 计算损失
            loss_vae, loss_gan = self._compute_losses(data, recon_batch, mu, logvar, 
                                                    disc_real, disc_fake, disc_factor)
            
            # 优化器步骤
            self.optimization_step(vae_optimizer, disc_optimizer, loss_vae, loss_gan)
            
            # 更新进度条
            pbar.set_postfix(VAE_Loss=loss_vae.item(), GAN_Loss=loss_gan.item())
            
            # 保存样本
            if batch_idx % 100 == 0:
                self._save_samples(data, recon_batch, epoch, batch_idx, save_dirs['results'])

    def setup_directories(self):
        """设置所需的目录结构"""
        dirs = {
            'results': os.path.join(self.config.BASE_PATH, "vae_results"),
            'checkpoints': os.path.join(self.config.BASE_PATH, "vae_checkpoints"),
            'model': os.path.join(self.config.BASE_PATH, "vae_model")
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return dirs