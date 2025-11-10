import os
import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import math
import numpy as np
from scene.cameras import Camera
from gaussian_renderer import render,render_adjust
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json
import cv2
from scipy.signal import find_peaks_cwt
import torch.nn.functional as F
from utils.simunet import SimpleUNet1D
def load_camera(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)

def ricker_wavelet(points, a, device='cuda'):
    x = torch.arange(-(points // 2), points // 2 + 1, dtype=torch.float32, device=device)
    x = x / a
    wavelet = (1 - x**2) * torch.exp(-x**2 / 2)
    return wavelet

def cwt_conv1d(signals, kernels):
    signals=signals.unsqueeze(1)
    cwt_matrix = F.conv1d(signals, kernels, padding=kernels.shape[2] // 2)  # (batch_size, n_scales, length)
    return cwt_matrix

def find_local_maxima(cwt_result, threshold=0.5):
    a=torch.roll(cwt_result,-1, dims=2)
    a[...,-1]=0
    b=torch.roll(cwt_result,+1, dims=2)
    b[...,0]=0
    cwt_result=torch.sum(((cwt_result<=a) & (cwt_result<=b) & ((cwt_result<b) | (cwt_result<a))), dim=1)>cwt_result.shape[1]*threshold
    return cwt_result
def find_local_maxima_single(cwt_result):
    a=torch.roll(cwt_result,-1, dims=2)
    a[...,-1]=0
    b=torch.roll(cwt_result,+1, dims=2)
    b[...,0]=0
    cwt_result=(cwt_result<=a) & (cwt_result<=b) & ((cwt_result<b) | (cwt_result<a))
    return cwt_result
def generate_signed_gaussian_kernel(radius: int, inner_radis: float, device=None, dtype=torch.float32):
    length = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    gauss = torch.exp(-(x ** 2) / (2 * inner_radis ** 2))
    gauss /= gauss.sum()
    kernel = gauss.clone()
    kernel[radius] = 0
    kernel[:radius] *= -1
    kernel[radius+1:] *= 1
    #kernel[radius-inner_radis:radius] *= -1
    #kernel[:radius-inner_radis]=0
    #kernel[radius+1:radius+1+inner_radis] *= 1
    #kernel[radius+1+inner_radis:]=0
    return kernel

def generate_multiple_kernels(radius: int, inner_radis, device=None, dtype=torch.float32):
    """
    sigmas: list or 1D tensor of sigma值
    返回形状 (len(sigmas), kernel_len)
    """
    kernels = []
    for inner_radi in inner_radis:
        k = generate_signed_gaussian_kernel(radius, inner_radi, device=device, dtype=dtype)
        kernels.append(k)
    return torch.stack(kernels, dim=0)

def conv1d_multiple_signed_gaussian_edge_extend(input_tensor: torch.Tensor, radius: int, inner_radis):
    """
    多核卷积，头尾超出部分用signal边界值扩展（replicate填充），无截断
    输入：
        input_tensor: (batch, length)
        radius: int, 核半径
        sigmas: list或tensor，多个核的sigma
    输出：
        (batch, num_kernels, length)
    """
    batch, length = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype

    # 产生多个kernel
    kernels = generate_multiple_kernels(radius, inner_radis, device=device, dtype=dtype)
    num_kernels, kernel_len = kernels.shape

    kernels = kernels.flip(dims=[1]).unsqueeze(1)  # (num_kernels, 1, kernel_len)
    x = input_tensor.unsqueeze(1)  # (batch, 1, length)

    # replicate填充，扩展边界值
    x_padded = F.pad(x, (radius, radius), mode='replicate')

    output = F.conv1d(x_padded, kernels, padding=0)
    return output
def derivative_1d(input_tensor):
    """
    对形状为 (N, L) 的 tensor 沿 L 维度做一维求导（差分）操作
    保持维度不变，padding='same'效果。
    """
    N, L = input_tensor.shape
    # Reshape为(N, 1, L)，方便用conv1d
    x = input_tensor.unsqueeze(1)  # (N, 1, L)

    # 差分核，简单的一阶导数，定义为 [-1, 0, 1]/2 可以模拟数值导数(central difference)
    kernel = torch.tensor([-1, 0, 1], dtype=torch.float32).view(1, 1, 3) / 2.0  # (out_channels, in_channels, kernel_size)

    # 卷积，padding=1 保持长度不变
    # 注意：kernel 需与 input_tensor 在同一设备
    kernel = kernel.to(input_tensor.device)

    # stride=1, padding=1 实现 same padding
    out = F.conv1d(x, kernel, padding=1)  # (N, 1, L)

    return out.squeeze(1)  # (N, L)

def gaussian_kernel_1d(kernel_size, sigma, device):
    """
    生成一维高斯卷积核（kernel_size必须是奇数）
    """
    assert kernel_size % 2 == 1, "kernel size must be odd"
    center = kernel_size // 2
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - center
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss /= gauss.sum()
    return gauss.view(1, 1, kernel_size)

def multi_gaussian_smooth(input_tensor, sigmas, kernel_size=21):
    """
    对input_tensor使用多尺度高斯滤波，返回形状 (N, len(sigmas), L)
    input_tensor: (N, L)
    sigmas: list or tensor of sigma值
    """
    N, L = input_tensor.shape
    x = input_tensor.unsqueeze(1)  # (N,1,L)
    device = input_tensor.device

    smoothed_outputs = []
    for sigma in sigmas:
        kernel = gaussian_kernel_1d(kernel_size, sigma, device)  # (1,1,kernel_size)
        padding = kernel_size // 2
        out = F.conv1d(x, kernel, padding=padding)  # (N,1,L)
        smoothed_outputs.append(out)

    # 拼接所有滤波结果 (N, len(sigmas), L)
    return torch.cat(smoothed_outputs, dim=1)

def extract_mesh(dataset, pipe, checkpoint_iterations=None):
    widths=range(20,30)
    kernel_size=31
    kernels=[]
    #simunet = SimpleUNet1D().cuda()   # 创建同结构模型（确保这段代码与你训练时模型定义一致）
    #simunet.load_state_dict(torch.load("unet_peak_detection.pth"))
    #simunet.eval()  # 切换到评估模式（关闭dropout，batchnorm等训练行为）
    #for scale in widths:
    #    kernel = ricker_wavelet(kernel_size, scale).flip(0)  # 翻转kernel用于卷积
    #    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,kernel_size)
    #    kernels.append(kernel)
    #kernels = torch.cat(kernels, dim=0)  # (n_scales,1,kernel_size)
    gaussians = GaussianModel(dataset.sh_degree)
    output_path = os.path.join(dataset.model_path,"point_cloud")
    iteration = 0
    if checkpoint_iterations is None:
        for folder_name in os.listdir(output_path):
            iteration= max(iteration,int(folder_name.split('_')[1]))
    else:
        iteration = checkpoint_iterations
    output_path = os.path.join(output_path,"iteration_"+str(iteration),"point_cloud.ply")

    gaussians.load_ply(output_path)
    print(f'Loaded gaussians from {output_path}')
    
    kernel_size = dataset.kernel_size
    
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_cam_list = load_camera(dataset)

    depth_list = []
    color_list = []
    alpha_thres = 0.5
    opa_multi=np.arange(0.005,0.005*255+0.005,0.005)
    for viewpoint_cam in viewpoint_cam_list:
        # Rendering offscreen from that camera 
        depthes = render_adjust(viewpoint_cam, gaussians, pipe, background, kernel_size,opas=opa_multi)
        depthes=torch.cat(depthes)
        
        #cv2.imwrite("depth0.png",depthes[30].cpu().numpy()*50)

        #print(depthes[30,200,200])
        #cv2.imwrite("depth1.png",depthes[200].cpu().numpy()*50)

        #print(depthes[200,200,200])
        #input()
        #x=depthes.reshape(len(opa_multi),-1).permute(1,0)
        #eps = 1e-8
        #min_vals = x.min(dim=1, keepdim=True).values   # 每行最小值，shape=(N_row,1)
        #max_vals = x.max(dim=1, keepdim=True).values   # 每行最大值，shape=(N_row,1)

        #x_norm = (x - min_vals) / (max_vals - min_vals + eps)  # 行归一化，shape不变
        #print(simunet(x_norm.squeeze(0).unsqueeze(1)).shape)
        #prob=simunet(x_norm.squeeze(0).unsqueeze(1)).squeeze(1).permute(1,0).reshape(depthes.shape)[:,200,200]
        #print(1)
        #print(F.sigmoid(prob)>0.5)
        plt.cla()
        plt.hist(depthes.cpu()[:,200,200])
        plt.savefig("hist.png")
        #input()
        #densities=conv1d_multiple_signed_gaussian_edge_extend(depthes.reshape(len(opa_multi),-1).permute(1,0),8,inner_radis=[3])
        d1d=derivative_1d(depthes.reshape(len(opa_multi),-1).permute(1,0)).abs()
        densities=multi_gaussian_smooth(d1d,[7.0],kernel_size=51)
        #print(densities.shape)
        #print(d1d.shape)
        #cv2.imwrite("depth.png",depthes[199].cpu().numpy()*50)
        height=250
        """
        for i in range(densities.shape[1]):
            plt.cla()
            plt.plot(range(depthes.shape[0]),densities[:,i,:].permute(1,0).reshape(depthes.shape)[:,200,height].cpu())
            plt.savefig("plt.png")
        #    input()
        """
        local_density_max=find_local_maxima_single(densities).squeeze(1)
        bool_tensor = local_density_max
        value_tensor = depthes.reshape(len(opa_multi),-1).permute(1,0)
        threshold=0.001
        # 计算相邻元素差距
        diff = torch.zeros_like(value_tensor, dtype=torch.float32)

        # 只在当前位置和前一位置都是True时才计算数值差
        mask = bool_tensor[:, 1:] & bool_tensor[:, :-1]
        diff[:, 1:][mask] = torch.abs(value_tensor[:, 1:][mask] - value_tensor[:, :-1][mask])

        # 判断差距是否小于阈值
        small_diff = (diff < threshold)

        # 确定组起点：
        # 当前位置为True，且：
        #   1) 如果前一个位置是False，则新组起点
        #   2) 如果前一个位置是True但差距大于阈值，也是新组起点
        # 用公式：组起点 = 当前True & （前一个False 或 差距大于等于阈值）
        prev_bool = torch.zeros_like(bool_tensor, dtype=torch.bool)
        prev_bool[:, 1:] = bool_tensor[:, :-1]

        group_start = bool_tensor & (~(prev_bool & small_diff))

        # cumsum给组编号（从1开始，减1变成0开始编号）
        seq_number = group_start.cumsum(dim=1) - 1

        # 非True位置标记为-1
        seq_number = seq_number.masked_fill(~bool_tensor, -1)
        valid_mask = (seq_number >= 0)

        # 创建累加和tensor
        max_group = 4
        N=depthes.shape[1]*depthes.shape[2]
        seq_fixed = seq_number.clone()
        seq_fixed[seq_fixed < 0] = 999  # 虚拟大组号，用来排除
        seq_clamped = seq_fixed.clamp(max=max_group)

        # 生成有效mask（元组非无效组）
        valid_mask = (seq_fixed != 999)

        # 创建累加和tensor，5组
        sum_per_group = torch.zeros((N, max_group + 1), dtype=value_tensor.dtype).cuda()
        count_per_group = torch.zeros((N, max_group + 1), dtype=value_tensor.dtype).cuda()

        # scatter加权累积
        sum_per_group.scatter_add_(1, seq_clamped, value_tensor * valid_mask)
        count_per_group.scatter_add_(1, seq_clamped, valid_mask.to(value_tensor.dtype))

        count_per_group = count_per_group.clamp(min=1)

        mean_per_group = sum_per_group / count_per_group

        print(mean_per_group.shape)
        print(depthes.shape)
        cv2.imwrite("depth.png",mean_per_group[...,1].reshape(depthes.shape[1],depthes.shape[2]).permute(1,0).cpu().numpy()*50)
        #indexes0=find_peaks_cwt(depthes[:,200,200].cpu(), np.arange(15, 25))
        #print(depthes[50,200,height])
        #print(depthes[100,200,height])
        #print(depthes[150,200,height])
        input()
        #input()
        #indexes0=find_peaks_cwt(depthes[:,200,200].cpu(), np.arange(15, 25))
        #print(depthes[:,200,200][indexes0])
        #cwt_result=cwt_conv1d(depthes.reshape(len(opa_multi),-1).permute(1,0),kernels)
        #print(cwt_result.shape)
        #local_maxima=find_local_maxima(cwt_result)
        #local_maxima=local_maxima.permute(1,0).reshape(depthes.shape)
        #indexes1=local_maxima[:,200,200]
        #print(depthes[:,200,200][indexes1])
        #plt.savefig("hist.png")
        #input()
        """
        for depth in depthes:
            cv2.imwrite("depth.png",depth.squeeze(0).cpu().numpy()*50)
            plt.cla()
            input()
        """
        if viewpoint_cam.gt_mask is not None:
            depth[(viewpoint_cam.gt_mask < 0.5)] = 0
        #depth[render_pkg["mask"]<alpha_thres] = 0
        #depth_list.append(depth[0].cpu().numpy())

    torch.cuda.empty_cache()
    voxel_size = 0.002
    o3d_device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'),
                                            attr_dtypes=(o3c.float32,
                                                         o3c.float32,
                                                         o3c.float32),
                                            attr_channels=((1), (1), (3)),
                                            voxel_size=voxel_size,
                                            block_resolution=16,
                                            block_count=50000,
                                            device=o3d_device)
    for color, depth, viewpoint_cam in zip(color_list, depth_list, viewpoint_cam_list):
        depth = o3d.t.geometry.Image(depth)
        depth = depth.to(o3d_device)
        color = o3d.t.geometry.Image(color)
        color = color.to(o3d_device)
        W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
        fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
        fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
        intrinsic = np.array([[fx,0,float(W)/2],[0,fy,float(H)/2],[0,0,1]],dtype=np.float64)
        intrinsic = o3d.core.Tensor(intrinsic)
        extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
        frustum_block_coords = vbg.compute_unique_block_coordinates(
                                                                        depth, 
                                                                        intrinsic,
                                                                        extrinsic, 
                                                                        1.0, 8.0
                                                                    )
        vbg.integrate(
                        frustum_block_coords, 
                        depth, 
                        color,
                        intrinsic,
                        extrinsic,  
                        1.0, 8.0
                    )

    mesh = vbg.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path,"recon.ply"),mesh.to_legacy())
    print("done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)
        
        
    
    