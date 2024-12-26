import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from scipy import linalg
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from VAE import VAE, ModelConfig

class FIDCalculator:
    def __init__(self, device='cuda'):
        self.inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.eval()
        self.inception.fc = torch.nn.Identity()
        self.inception.to(device)
        self.device = device
        
        # 修改预处理步骤
        self.transform = transforms.Compose([
            transforms.Resize(299, antialias=True),  # 从32x32放大到299x299
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
    
    def get_activations(self, images):
        batch_size = 32
        n_samples = len(images)
        n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)
        
        pred_arr = np.empty((n_samples, 2048))
        
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            batch = images[start:end].to(self.device)
            batch = self.transform(batch)
            
            with torch.no_grad():
                pred = self.inception(batch)[0]
            
            pred_arr[start:end] = pred.cpu().numpy()
            
        return pred_arr
    
    def calculate_fid(self, act1, act2):
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid



def evaluate_vaes(model_paths, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CIFAR-10特定的transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # 不需要resize，保持原始32x32分辨率
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 加载所有模型
    vae_models = {}
    
    for name, path in model_paths.items():
        print(f"Loading {name} from {path}")
        # 初始化模型
        model = VAE(modelconf.LATENT_CHANNEL, modelconf.DOWNSAMPLE)
        # 加载权重
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        vae_models[name] = model
    
    # 初始化FID计算器
    fid_calculator = FIDCalculator(device)
    
    # 收集原始图像
    print("Collecting original images...")
    original_images = []
    for images, _ in tqdm(test_loader):
        original_images.append(images)
    original_images = torch.cat(original_images, dim=0)
    
    # 获取原始图像的Inception特征
    print("Computing features for original images...")
    original_acts = fid_calculator.get_activations(original_images)
    
    results = {}
    
    # 评估每个VAE模型
    for model_name, vae in vae_models.items():
        print(f"\nEvaluating {model_name}...")
        vae.eval()
        vae.to(device)
        
        reconstructed_images = []
        
        with torch.no_grad():
            for images, _ in tqdm(test_loader):
                images = images.to(device)
                recon, _, _ = vae(images)
                reconstructed_images.append(recon.cpu())
                
        reconstructed_images = torch.cat(reconstructed_images, dim=0)
        
        print(f"Computing features for reconstructed images from {model_name}...")
        recon_acts = fid_calculator.get_activations(reconstructed_images)
        
        # 计算FID
        fid_score = fid_calculator.calculate_fid(original_acts, recon_acts)
        results[model_name] = fid_score
        
        print(f"{model_name} FID Score: {fid_score:.2f}")
        
        # 保存一些重建图像样本
        save_reconstruction_samples(original_images[:8], 
                                 reconstructed_images[:8], 
                                 model_name)
        
    return results

def save_reconstruction_samples(original, reconstructed, model_name):
    """保存原始图像和重建图像的对比"""
    plt.figure(figsize=(12, 6))
    
    # 显示原始图像
    for i in range(8):
        plt.subplot(2, 8, i + 1)
        plt.imshow(original[i].permute(1, 2, 0).numpy())
        plt.axis('off')
        if i == 0:
            plt.title('Original')
    
    # 显示重建图像
    for i in range(8):
        plt.subplot(2, 8, i + 9)
        plt.imshow(reconstructed[i].permute(1, 2, 0).numpy())
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(f'reconstruction_samples_{model_name}.png')
    plt.close()

def visualize_results(results):
    """可视化FID得分"""
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    scores = list(results.values())
    
    plt.bar(names, scores)
    plt.title('FID Scores for Different VAE Models')
    plt.xlabel('Model')
    plt.ylabel('FID Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fid_comparison.png')
    plt.close()


if __name__ == "__main__":
    # 定义模型路径
    model_paths = {
        'VAE_model_L2_recon': 'trained_VAE_models/recon/final_vae_model.pt',
        'VAE_model_L2_recon_per': 'trained_VAE_models/recon_per/final_vae_model.pt',
        'VAE_model_L2_recon_per_gan': 'trained_VAE_models/recon_per_gan/final_vae_model.pt'
    }

    modelconf = ModelConfig()
    
  
    
    # 运行评估
    results = evaluate_vaes(model_paths)
    
    # 保存结果
    import json
    with open('fid_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 可视化结果
    visualize_results(results)
    
    # 分析结果
    best_model = min(results.items(), key=lambda x: x[1])
    print(f"\nBest model: {best_model[0]} with FID: {best_model[1]:.2f}")