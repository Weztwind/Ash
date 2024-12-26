import torch
import os
import torch.nn.functional as F
from torchvision.utils import save_image


from tqdm import tqdm

class VAETrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def vae_compute_losses(self, data, recon_batch, mu, logvar, disc_fake):
        """计算vae损失函数"""
        loss_perceptual = self.model.perceptual_loss(data, recon_batch)#一个batch的损失
        loss_perceptual = loss_perceptual.mean()
        # print(f"Raw perceptual loss (before mean): {loss_perceptual}")  # 打印原始感知损失
        # print(f"Perceptual loss (after mean): {loss_perceptual.item()}")  # 打印mean后的感知损失
        

        loss_recon = torch.abs(data - recon_batch)
        loss_recon =  loss_recon.mean()

        
        # loss_recon = loss_recon = ((data - recon_batch) ** 2)
        # loss_recon =  loss_recon.mean()

        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())###实验时直接删掉这个，否则训练不出来
        loss_kl = loss_kl.mean()

        loss_per_recon_kl = loss_perceptual + loss_recon + 0 * loss_kl
        loss_g = -torch.mean(disc_fake)#希望disc_fake尽可能大
        
        λ = self.model.calculate_lambda(loss_per_recon_kl, loss_g)
        loss_vae_ganpart = self.config.DISCFACTOR * λ * loss_g
        loss_vae = loss_per_recon_kl + self.config.DISCFACTOR * λ * loss_g
        
        return loss_vae, loss_perceptual, loss_recon, loss_vae_ganpart, λ, loss_g
    
    def vae_compute_losses_without_gan(self, data, recon_batch, mu, logvar):
        """计算vae损失函数"""
        loss_perceptual = self.model.perceptual_loss(data, recon_batch)
        loss_perceptual = loss_perceptual.mean()

        # loss_recon = torch.abs(data - recon_batch)
        # loss_recon =  loss_recon.mean()

        loss_recon = loss_recon = ((data - recon_batch) ** 2)
        loss_recon =  loss_recon.mean()

        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())###实验时直接删掉这个，否则训练不出来
        loss_kl = loss_kl.mean()
        loss_vae = loss_perceptual + loss_recon + 0 * loss_kl
        
        return loss_vae, loss_perceptual, loss_recon
    
    def vae_compute_losses_without_gan_and_perceptual(self, data, recon_batch, mu, logvar):
        """计算vae损失函数"""
        loss_recon = torch.abs(data - recon_batch)
        loss_recon =  loss_recon.mean()

        # loss_recon = loss_recon = ((data - recon_batch) ** 2)
        # loss_recon =  loss_recon.mean()

        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())###实验时直接删掉这个，否则训练不出来
        loss_kl = loss_kl.mean()


        loss_vae = loss_recon + 0 * loss_kl
        
        return loss_vae
    
    def gan_compute_losses(self, disc_real, disc_fake, disc_factor):
        """计算gan损失函数"""
        
        loss_d_real = torch.mean(F.relu(1. - disc_real))#希望disc_real大于1
        loss_d_fake = torch.mean(F.relu(1. + disc_fake))#希望disc_fake小于-1
        loss_gan = disc_factor * 0.5 * (loss_d_real + loss_d_fake)
        
        return loss_gan
    
    def vae_optimization_step(self, vae_optimizer, loss_vae):
        vae_optimizer.zero_grad()
        loss_vae.backward()
        vae_optimizer.step()

    def gan_optimization_step(self, disc_optimizer, loss_gan):
        disc_optimizer.zero_grad()
        loss_gan.backward()
        disc_optimizer.step()
        
    def _save_samples(self, data, recon_batch, epoch, batch_idx, save_dir):
        """保存重构样本"""
        with torch.no_grad():
            comparison = torch.cat((data[:4], recon_batch[:4]))
            save_image((comparison + 1)/2 , 
                      os.path.join(save_dir, f"recon_epoch{epoch+1}_batch{batch_idx}.png"), 
                      nrow=4)
    
    def _save_checkpoint(self, epoch, save_dir, psnr_history=None):
        """保存模型检查点
        Args:
            epoch: 当前轮次
            save_dir: 保存路径
            psnr_history: PSNR历史记录字典
        """
        # 保存模型状态和PSNR历史
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'psnr_history': psnr_history if psnr_history is not None else {}
        }
        
        checkpoint_path = os.path.join(save_dir, f"vae_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def calculate_psnr(self, img1, img2):
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        mse = F.mse_loss(img1, img2, reduction='mean')
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr
        
    def train_epoch(self, train_loader, vae_optimizer, disc_optimizer, epoch, total_epochs, save_dirs):
        """训练一个epoch"""
        steps_per_epoch = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, 
                   desc=f"Epoch {epoch + 1}/{total_epochs}")
        
        # 用于计算整个epoch的平均PSNR
        total_train_psnr = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in pbar:
            data = data.to(self.config.DEVICE, dtype=torch.float)
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            disc_fake = self.model.discriminator_forward(recon_batch)

            # 计算当前batch的PSNR
            batch_psnr = self.calculate_psnr(data, recon_batch)
            total_train_psnr += batch_psnr.item()
            num_batches += 1
            
            # 计算损失
            # loss_vae = self.vae_compute_losses_without_gan_and_perceptual(data, recon_batch, mu, logvar)
            # loss_vae, loss_perceptual, loss_recon = self.vae_compute_losses_without_gan(data, recon_batch, mu, logvar)
            loss_vae, loss_perceptual, loss_recon, loss_vae_ganpart, lambda_gan, loss_g = self.vae_compute_losses(data, recon_batch, mu, logvar, disc_fake)
            
            # 优化器步骤
            self.vae_optimization_step(vae_optimizer, loss_vae)

            with torch.no_grad():
                recon_batch_new = self.model(data)[0]


            disc_real = self.model.discriminator_forward(data)
            disc_fake = self.model.discriminator_forward(recon_batch_new.detach())
            # 计算判别器因子
            disc_factor = self.model.adopt_weight(self.config.DISCFACTOR, 
                                                epoch * steps_per_epoch + batch_idx,
                                                threshold=self.config.DISC_START)

            # 计算损失
            loss_gan = self.gan_compute_losses(disc_real, disc_fake, disc_factor)
                                                    
            if epoch > self.config.DISC_START:
                self.gan_optimization_step(disc_optimizer, loss_gan)
                # if batch_idx % 30 == 0:
                #     # 判别器步骤
                #     self.gan_optimization_step(disc_optimizer, loss_gan)

        
            # 更新进度条，添加PSNR信息
            pbar.set_postfix({
                'GAN_Loss': f'{loss_gan.item():.4f}',
                'disc_real': f'{torch.mean(disc_real).item():.4f}',
                'disc_fake': f'{torch.mean(disc_fake).item():.4f}',
                'VAE_Loss': f'{loss_vae.item():.4f}',
                'VAE_Loss_perceptual':f'{loss_perceptual.item():.4f}',
                'VAE_Loss_recon':f'{loss_recon.item():.4f}',
                'VAE_Loss_gan_part':f'{loss_vae_ganpart.item():.4f}',
                'lambda_gan':f'{lambda_gan.item():.4f}',
                'loss_g':f'{loss_g.item():.4f}',
                'train_PSNR': f'{batch_psnr.item():.2f}'
            })

            # pbar.set_postfix({
            #     'VAE_Loss': f'{loss_vae.item():.4f}',
            #     'VAE_Loss_perceptual':f'{loss_perceptual.item():.4f}',
            #     'VAE_Loss_recon':f'{loss_recon.item():.4f}',
            #     'train_PSNR': f'{batch_psnr.item():.2f}'
            # })
            
            # 保存样本
            if batch_idx % 100 == 0:
                self._save_samples(data, recon_batch, epoch, batch_idx, save_dirs['results'])

        # 计算并保存整个epoch的平均PSNR
        epoch_train_avg_psnr = total_train_psnr / num_batches
        psnr_summary_path = os.path.join(save_dirs['results'], 'epoch_train_psnr_summary.txt')
        with open(psnr_summary_path, 'a') as f:
            f.write(f'Epoch {epoch+1}, Average train PSNR: {epoch_train_avg_psnr:.2f}\n')

        return epoch_train_avg_psnr
    
    def test_epoch(self, test_loader, epoch, total_epochs, save_dirs):
        self.model.eval()
        total_test_psnr = 0.0
        num_test_batches = 0
        
        # 创建测试进度条
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{total_epochs} (Test)")
        
        with torch.no_grad():
            for test_data, _ in test_pbar:
                test_data = test_data.to(self.config.DEVICE, dtype=torch.float)
                
                # Forward pass
                test_recon, test_mu, test_logvar = self.model(test_data)
                # test_disc_fake = self.model.discriminator_forward(test_recon)
                
                # 计算测试集PSNR
                test_batch_psnr = self.calculate_psnr(test_data, test_recon)
                total_test_psnr += test_batch_psnr.item()
                
                # # 计算测试集损失
                # test_loss_vae, _, _, _, _ = self.vae_compute_losses(
                #     test_data, test_recon, test_mu, test_logvar, test_disc_fake)
                # total_test_vae_loss += test_loss_vae.item()
                
                # # 计算测试集GAN损失
                # test_disc_real = self.model.discriminator_forward(test_data)
                # test_loss_gan = self.gan_compute_losses(test_disc_real, test_disc_fake, 1.0)
                # total_test_gan_loss += test_loss_gan.item()
                
                num_test_batches += 1
                
                # 更新测试进度条
                test_pbar.set_postfix({
                    'Test_PSNR': f'{test_batch_psnr.item():.2f}'
                })
        # 计算并保存整个epoch的平均PSNR
        epoch_test_avg_psnr = total_test_psnr / num_test_batches
        psnr_summary_path = os.path.join(save_dirs['results'], 'epoch_test_psnr_summary.txt')
        with open(psnr_summary_path, 'a') as f:
            f.write(f'Epoch {epoch+1}, Average test PSNR: {epoch_test_avg_psnr:.2f}\n')
        
        return epoch_test_avg_psnr


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