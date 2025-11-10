import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def generate_gmm_signals_with_edge_cases(batch_size, length,
                                         num_peaks_range=(0,3),
                                         sigma_range=(1.0,3.0),
                                         amplitude_range=(0.5,1.5),
                                         noise_std=0.05,
                                         edge_peak_prob=0.3,  # 有30%概率在边界截断峰
                                         safe_region_ratio=0.2, # 安全区域边界比例
                                         device='cpu'):
    """
    生成多峰信号，部分峰完整，部分峰靠近边界截断，信号加噪声
    
    edge_peak_prob：每个峰独立决定是否靠近边界截断
    safe_region_ratio：如果不截断，峰均值在[safe_region_ratio, 1 - safe_region_ratio]区间
    """
    signals = torch.zeros(batch_size, length, device=device)
    peak_maps = torch.zeros(batch_size, length, device=device)
    x = torch.arange(length, device=device).float()

    for b in range(batch_size):
        num_peaks = random.randint(*num_peaks_range)
        for _ in range(num_peaks):
            sigma = random.uniform(*sigma_range)
            amplitude = random.uniform(*amplitude_range)
            
            if random.random() < edge_peak_prob:
                # 边界截断峰
                # 选择靠近左边界还是右边界
                if random.random() < 0.5:
                    # 靠近左边界，峰均值在[0, safe_region_ratio * length]
                    mean = random.uniform(0, safe_region_ratio * length)
                else:
                    # 靠近右边界，峰均值在[(1 - safe_region_ratio)*length, length]
                    mean = random.uniform((1 - safe_region_ratio) * length, length)
            else:
                # 安全峰，均值在 [safe_region_ratio*length, (1 - safe_region_ratio)*length]
                mean = random.uniform(safe_region_ratio * length, (1 - safe_region_ratio) * length)

            gauss = torch.exp(- ((x - mean) ** 2) / (2 * sigma ** 2))
            signals[b] += amplitude * gauss
            # 对峰标签使用最大值叠加
            peak_maps[b] = torch.max(peak_maps[b], gauss)

        # 加噪声
        noise = torch.randn(length, device=device) * noise_std
        signals[b] += noise
        # 归一化信号
        signals[b] = (signals[b] - signals[b].min()) / (signals[b].max() - signals[b].min() + 1e-8)

    return signals.unsqueeze(1), peak_maps.unsqueeze(1)


class SimpleUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, base_filters, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(nn.Conv1d(base_filters, base_filters * 2, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool1d(2)
        self.bottleneck = nn.Sequential(nn.Conv1d(base_filters * 2, base_filters * 4, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv1d(base_filters * 4, base_filters * 2, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose1d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv1d(base_filters * 2, base_filters, 3, padding=1), nn.ReLU())
        self.final = nn.Conv1d(base_filters, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x) 
        p1 = self.pool1(e1) 
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)