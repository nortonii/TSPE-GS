import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils.simunet import SimpleUNet1D
def generate_gmm_signals_with_edge_cases_hard_label(batch_size, length,
                                         num_peaks_range=(0,10),
                                         sigma_range=(0.5,4.0),
                                         amplitude_range=(0.5,1.5),
                                         noise_std=0.05,
                                         edge_peak_prob=0.3,
                                         safe_region_ratio=0.2,
                                         device='cpu'):
    signals = torch.zeros(batch_size, length, device=device)
    peak_maps = torch.zeros(batch_size, length, device=device)
    x = torch.arange(length, device=device).float()

    for b in range(batch_size):
        num_peaks = random.randint(*num_peaks_range)
        for _ in range(num_peaks):
            sigma = random.uniform(*sigma_range)
            amplitude = random.uniform(*amplitude_range)
            
            if random.random() < edge_peak_prob:
                if random.random() < 0.5:
                    mean = random.uniform(0, safe_region_ratio * length)
                else:
                    mean = random.uniform((1 - safe_region_ratio) * length, length)
            else:
                mean = random.uniform(safe_region_ratio * length, (1 - safe_region_ratio) * length)

            gauss = torch.exp(- ((x - mean) ** 2) / (2 * sigma ** 2))
            signals[b] += amplitude * gauss

            # 硬标签：峰中心对应索引置1，索引必须是整数且有效范围内
            peak_center = int(round(mean))
            if 0 <= peak_center < length:
                peak_maps[b, peak_center] = 1.0

        noise = torch.randn(length, device=device) * noise_std
        signals[b] += noise
        signals[b] = (signals[b] - signals[b].min()) / (signals[b].max() - signals[b].min() + 1e-8)

    return signals.unsqueeze(1), peak_maps.unsqueeze(1)


if __name__ == "__main__":
    # 假设 SimpleUNet1D 和训练代码复用之前的代码块

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    epochs = 2000
    batch_size = 16
    length = 256
    

    #pos_weight = torch.tensor([128.0], device=device)  # 根据需要调整权重
    criterion = nn.BCEWithLogitsLoss()#pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        signals, peak_maps = generate_gmm_signals_with_edge_cases_hard_label(batch_size, length, device=device)
        logits = model(signals)  # 这里输出 logits，大小应同 peak_maps
        
        # logits += 1.0  # 如果需要可以调试加上这一句提升正样本概率
        
        loss = criterion(logits, peak_maps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    # 假设你的模型实例是 model
torch.save(model.state_dict(), "unet_peak_detection.pth")