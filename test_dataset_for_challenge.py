from dataset_challenge import get_train_val_test_loaders, DogsDataset, ImageStandardizer

import torch
import numpy as np
import matplotlib.pyplot as plt

def unnormalize_chw(x: torch.Tensor, mean, std) -> np.ndarray:
    """x: (C,H,W) tensor normalized by (mean, std) in [0,1] space.
       mean/std: 1D array-like of length 3 in [0,1].
       return: (H,W,C) float numpy in [0,1]."""
    if isinstance(mean, np.ndarray): mean = mean.tolist()
    if isinstance(std,  np.ndarray): std  = std.tolist()
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=x.dtype, device=x.device).view(-1, 1, 1)
    x = x * std_t + mean_t
    x = torch.clamp(x, 0, 1)
    x = x.permute(1, 2, 0)  # CHW -> HWC
    return x.detach().cpu().numpy()

def show_batch(batch_imgs: torch.Tensor, batch_labels: torch.Tensor, mean, std, title: str, max_n: int = 8):
    """batch_imgs: (B,C,H,W), already normalized; labels: (B,)"""
    b = min(max_n, batch_imgs.size(0))
    fig, axes = plt.subplots(1, b, figsize=(2*b, 2))
    if b == 1:
        axes = [axes]
    for i in range(b):
        img_np = unnormalize_chw(batch_imgs[i], mean, std)
        axes[i].imshow(img_np)
        axes[i].set_title(str(int(batch_labels[i].item())))
        axes[i].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def main():
    task = "source"  # 和下面 tr_tmp 计算 mean/std 的 task 保持一致
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task=task,
        batch_size=32,
    )

    # 取一个训练 batch 看看形状
    train_X, train_y = next(iter(tr_loader))  # [B, C, H, W]
    print(train_X.shape)

    # 用同一 task 的原始训练图像计算 mean/std（单位: 0~255）
    tr_tmp = DogsDataset("train", task, transform=None)
    standardizer = ImageStandardizer()
    standardizer.fit(tr_tmp.X)

    # 转成 Normalize 使用的 0~1 空间
    mean = (standardizer.image_mean.squeeze() / 255.0).astype(np.float32)
    std  = (standardizer.image_std.squeeze()  / 255.0).astype(np.float32)

    image_dim = tr_tmp.X[0].shape[0]
    print(image_dim)
    print(f"Mean(0~255):\t{standardizer.image_mean}")
    print(f"Std(0~255): \t{standardizer.image_std}")

    # === 展示图像 ===
    # 训练集（含强数据增强）的一个 batch
    show_batch(train_X, train_y, mean, std, title="Train batch (augmented)")

    # 验证集（中心裁剪等轻度预处理）
    val_X, val_y = next(iter(va_loader))
    show_batch(val_X, val_y, mean, std, title="Val batch (center-crop + normalize)")

    # 测试集
    te_X, te_y = next(iter(te_loader))
    show_batch(te_X, te_y, mean, std, title="Test batch (center-crop + normalize)")

if __name__ == "__main__":
    main()
