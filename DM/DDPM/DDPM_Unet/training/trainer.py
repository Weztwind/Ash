import torch
from ..diffusion import forward_diffusion
from tqdm import tqdm
from torch.cuda import amp
from torchmetrics import MeanMetric
from .traincofig import BaseConfig, TrainingConfig

def train_one_epoch(model, sd, loader, optimizer, scaler, loss_fn, epoch, total_epochs, user_id=None , base_config=BaseConfig(), training_config=TrainingConfig()):
    """
    训练一个epoch的函数。

    Args:
        model: 要训练的模型。
        sd: 数据扩散器。
        loader: 数据加载器。
        optimizer: 优化器。
        scaler: 梯度缩放器,用于自动混合精度(AMP)训练。
        loss_fn: 损失函数。
        epoch: 当前的epoch数。
        base_config: 基本配置对象。
        training_config: 训练配置对象。

    Returns:
        mean_loss: 整个epoch的平均损失。
    """
    loss_record = MeanMetric()  # 创建一个用于记录平均损失的对象
    model.train()  # 将模型设置为训练模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:  # 创建一个进度条
         # 根据user_id是否存在设置不同的描述
        if user_id is not None:
            tq.set_description(f"Train user{user_id + 1} :: Epoch: {epoch}/{total_epochs}")
        else:
            tq.set_description(f"Train Epoch: {epoch}/{total_epochs}")

        for x0s, label in loader:  # 对数据加载器中的每一批数据进行处理
            tq.update(1)  # 更新进度条
            x0s = x0s.to(device)
            label = label.to(device)

            # 随机生成一批时间步
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)

            xts, gt_noise = forward_diffusion(sd, x0s, ts)  # 对输入数据进行前向扩散

            with amp.autocast():  # 使用自动混合精度(AMP)
                pred_noise = model(xts, label, ts)  # 使用模型预测噪声
                loss = loss_fn(gt_noise, pred_noise)  # 计算损失

            optimizer.zero_grad(set_to_none=True)  # 清零梯度
            scaler.scale(loss).backward()  # 对损失进行缩放,并计算梯度

            # scaler.unscale_(optimizer)  # 如果需要的话,可以对优化器进行梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)  # 更新优化器
            scaler.update()  # 更新scaler

            loss_value = loss.detach().item()  # 获取损失值
            loss_record.update(loss_value)  # 记录损失

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")  # 在进度条中显示当前批次的损失

        mean_loss = loss_record.compute().item()  # 计算整个epoch的平均损失

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")  # 在进度条中显示整个epoch的平均损失

    return mean_loss  # 返回整个epoch的平均损失
