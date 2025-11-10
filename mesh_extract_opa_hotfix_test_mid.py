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
from gaussian_renderer import render,render_threshold
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json

import torch
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import requests
import base64
import numpy as np
import io
from PIL import Image
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import torch
import numpy as np
import torch
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import subprocess
from scipy.signal import find_peaks, savgol_filter
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(0)
print(str(np.argmin([int(x.split()[2]) for x in result[:-1]])))
os.system('echo $CUDA_VISIBLE_DEVICES')

import cv2
import kornia
def compute_pdf(q):
#    p = torch.linspace(0, 1, len(q))  # [0, 1], 长度256
#    h = p[1] - p[0]  # = ~1/255

#    dQdp = torch.zeros_like(q)

#    # 内部点的中央差分
#    dQdp[1:-1] = (q[2:] - q[:-2]) / (2 * h)

    # 两端用前向和后向差分
#    dQdp[0] = (q[1] - q[0]) / h
#    dQdp[-1] = (q[-1] - q[-2]) / h

    # 为防止分母为0，加个小数
    eps = 1e-8
    dQdp=torch.diff(q)
    torch.clamp(-dQdp,eps)
    # 计算pdf
    pdf = 1.0 / (dQdp + eps)
    return pdf

# 计算每个样本所属最近中心的索引
def get_closest_point_indices_for_each_center(X, gmm):
    centers = gmm.means_  # shape: (k, n_features)
    closest_indices = []
    for index,center in enumerate(centers):
        weight = gmm.weights_[index]
        if weight < 0.2:  # 如果权重小于0.01
            continue
        # 计算所有点到当前中心的欧氏距离
        distances = np.linalg.norm(X - center, axis=1)
        # 找到距离最小点的索引
        closest_idx = np.argmin(distances)
        closest_indices.append(closest_idx)
    return closest_indices
from scipy.signal import find_peaks, savgol_filter
import torch
import numpy as np
from diptest import diptest
import tqdm
def dip_test_sample(tensor):
    data = tensor
    dip, p_value = diptest(data)
    return dip, p_value

all_d=[]
all_p=[]
import numpy as np
from scipy.stats import gaussian_kde

def batch_kde_peak_extraction(batch_imgs: np.ndarray, gt_mask: np.ndarray, mask_threshold=0.5, grid_points=1000):
    """
    对batch_imgs中的每个批次，选取gt_mask大于threshold的像素数据，
    对每个批次的数据做KDE估计，并提取峰值位置和峰值高度。
    
    参数：
    - batch_imgs: np.ndarray, 形状(512, ...)的batch数据
    - gt_mask: np.ndarray, 与batch_imgs单个样本空间形状对应的mask，形状与batch_imgs后半维相符
    - mask_threshold: float，阈值，默认0.5，用于筛选gt_mask
    - grid_points: int，KDE估计时的采样点数，默认1000
    
    返回：
    - peak_positions: np.ndarray, 长度等于batch大小，每个批次KDE峰值所在x坐标
    - peak_values: np.ndarray, 长度等于batch大小，每个批次KDE峰值对应概率密度值
    """
    batch_size = batch_imgs.shape[0]
    
    # 展平batch后面所有维度
    data = batch_imgs.reshape(batch_size, -1)
    mask = gt_mask.reshape(-1) > mask_threshold
    
    selected_data = data[:, mask]  # 筛选后形状 (batch_size, M)
    
    peak_positions = np.full(batch_size, np.nan, dtype=np.float32)
    peak_values = np.full(batch_size, np.nan, dtype=np.float32)
    
    for i in range(batch_size):
        values = selected_data[i]
        if values.size == 0:
            continue
        
        kde = gaussian_kde(values)
        x_grid = np.linspace(np.min(values), np.max(values), grid_points)
        density = kde(x_grid)
        
        peak_idx = np.argmax(density)
        peak_positions[i] = x_grid[peak_idx]
        peak_values[i] = density[peak_idx]
    
    return peak_positions, peak_values
def smooth_tensor_monotone_decreasing(input_tensor):
    arr = input_tensor.cpu().numpy()
    
    unique_vals = []
    boundaries = [0]
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            unique_vals.append(arr[i-1])
            boundaries.append(i)
    unique_vals.append(arr[-1])
    boundaries.append(len(arr))
    
    boundaries = torch.tensor(boundaries)
    unique_vals = torch.tensor(unique_vals)
    
    output = torch.zeros_like(input_tensor, dtype=torch.float32)
    
    # 插值，同时做单调递减约束
    for i in range(len(unique_vals)-1):
        start_idx = boundaries[i].item()
        end_idx = boundaries[i+1].item()
        length = end_idx - start_idx
        
        start_val = unique_vals[i].item()
        end_val = unique_vals[i+1].item()
        
        # 保证end_val不大于start_val，强制单调递减
        if end_val > start_val:
            end_val = start_val
        
        interp_values = torch.linspace(start_val, end_val, steps=length)
        output[start_idx:end_idx] = interp_values
    
    # 最后一组全部赋值为其值
    last_start = boundaries[-2].item()
    last_end = boundaries[-1].item()
    last_val = unique_vals[-1].item()
    # 如果最后值比前一组大，限制为前一组最后值
    if last_val > output[last_start - 1]:
        last_val = output[last_start - 1]
    output[last_start:last_end] = last_val
    
    # 后处理：强制全局单调递减，去除局部glitch
    for i in range(1, len(output)):
        if output[i] > output[i-1]:
            output[i] = output[i-1]
    
    return output


def mask_peak_and_filter_batches(batch_imgs: torch.Tensor,
                                gt_mask: torch.Tensor,
                                prom: float = 0.1):
    """
    1. 对每个像素512维信号找峰，峰值显著性prominence阈值过滤，非峰置0
    2. 对gt_mask<0.5像素，全部置0
    3. 移除所有批次中全零的切片

    Args:
        batch_imgs: [512, W, H]
        gt_mask: [W, H]
        prom: float, 峰值显著性阈值相对于峰值范围的比例（0-1）

    Returns:
        filtered_batch_imgs: [N, W, H], N = 原始非全零批次数
    """
    B, W, H = batch_imgs.shape
    output = torch.zeros_like(batch_imgs)
    
    arr = batch_imgs.cpu().numpy()       # (512, W, H)
    gt_np = gt_mask.cpu().numpy()        # (W, H)
    
    for x in range(W):
        for y in range(H):
            if gt_np[x, y] < 0.5:
                continue

            signal = arr[:, x, y]
            # 计算峰值显著性阈值
            prom_threshold = prom * (signal.max() - signal.min())
            if prom_threshold <= 0:
                # 信号全等值时避免错误，直接找所有峰（其实没峰）
                peaks, _ = np.array([]), {}
            else:
                peaks, properties = find_peaks(signal, prominence=prom_threshold)

            if peaks.size == 0:
                # 没找到满足显著性的峰，考虑保留最大值位置作为“峰”
                max_idx = np.argmax(signal)
                peaks = np.array([max_idx])

            output_peaks = np.zeros(B, dtype=signal.dtype)
            output_peaks[peaks] = signal[peaks]
            output[:, x, y] = torch.from_numpy(output_peaks)

    output = output.to(batch_imgs.device)

    # 检查每个批次是否全零，保留非全零批次索引
    nonzero_mask = ~(output.view(B, -1).abs().sum(dim=1) == 0)  # bool向量
    filtered_output = output[nonzero_mask]

    return filtered_output
def remove_adjacent_duplicates(batch_imgs: torch.Tensor, gt_mask, prom):
    #results = dip_test_batch(selected_data)

    #dips = np.array([r[0] for r in results])
    #p_values = np.array([r[1] for r in results])
    #print(batch_imgs[:,30,30])
    #print(smooth_tensor_monotone_decreasing(batch_imgs[:,30,30]))
    #plt.plot(range(512),smooth_tensor_monotone_decreasing(batch_imgs[:,30,30]).cpu().numpy())
    #plt.savefig("smooth.png")
    #exit()
    #results = batch_kde_peak_extraction(batch_imgs.cpu().numpy(),gt_mask.cpu().numpy())
    #print(results)
    #exit()

    s_sum1= batch_imgs.mean(dim=(1, 2))  # 计算每张图的平均值
    
    data_np= s_sum1.cpu().numpy()
    kde = gaussian_kde(data_np)
    plt.cla()
    value_range=data_np.max()-data_np.min()
    x_eval = np.linspace(data_np.min()-value_range*0.02, data_np.max(), 1000)
    density=kde(x_eval)
    
    density=density/density.sum()
    #print(density[304])
    #print(density[682])
    #print(density[999])
    #print("min",density.min())
    #print("max",density.max())
    #print(0.001*(density.max()-density.min()))
    #plt.plot(range(1000),density)
    
    peaks, pin = find_peaks(density,prominence  =prom*(density.max()-density.min()))  # 找到峰值

    #print(pin)
    #print(0.01*(density.max()-density.min()))
    #peaks=peaks[pin["prominences"]>0.01*(density.max()-density.min())]
    peak_positions = x_eval[peaks]
    closest_indices = []
    #print(peak_positions)
    #print(peaks)
    for pos in peak_positions:
        idx = np.argmin(np.abs(data_np - pos))
        closest_indices.append(idx)
    #print(closest_indices)
    #plt.savefig("hist.png")
    #input()
    #print(peaks)
    #print(closest_indices)
    #input()
    #input()
    # 1步差分：长度变成 N-1
    """
    diff_1 = batch_imgs[1:] - batch_imgs[:-1]  # 9个差分
    diff_2 = batch_imgs[2:] - batch_imgs[:-2]  # 8个差分
    diff_3 = batch_imgs[3:] - batch_imgs[:-3]  # 8个差分
    diff_4 = batch_imgs[4:] - batch_imgs[:-4]  # 8个差分

    # 对齐到原长度 N = batch_imgs.shape[0]，用0填充尾部
    pad_1 = torch.zeros_like(batch_imgs[0:1])  # 形状和一个元素一样
    diff_1_padded = torch.cat([diff_1, pad_1], dim=0)  # 变成10个元素

    pad_2 = torch.zeros_like(batch_imgs[0:2])  # 2个元素零
    diff_2_padded = torch.cat([diff_2, pad_2], dim=0)  # 变成10个元素

    pad_3 = torch.zeros_like(batch_imgs[0:3])  # 2个元素零
    diff_3_padded = torch.cat([diff_3, pad_3], dim=0)  # 变成10个元素

    pad_4 = torch.zeros_like(batch_imgs[0:4])  # 2个元素零
    diff_4_padded = torch.cat([diff_4, pad_4], dim=0)  # 变成10个元素

    # 相加
    maskes = torch.abs(diff_1_padded) + torch.abs(diff_2_padded)+torch.abs(diff_3_padded)+torch.abs(diff_4_padded)  # 10个元素，与原batch_imgs对齐

    best_k, bics, best_gmm=(fit_gmm_and_select_k(s_sum1.unsqueeze(1).cpu()))
    
    index0=get_closest_point_indices_for_each_center(s_sum1.unsqueeze(1).cpu(),best_gmm)
    
    for index in index0:
        if index<index0[np.argmax(index0)]:
            mask=(maskes[index]>0.0)
            batch_imgs[index][mask]= 0
        print(index)
        cv2.imwrite("depth.png",batch_imgs[index].cpu().numpy()*50)
        input()
    """
    #for index in closest_indices:
    #    cv2.imwrite("depth.png",batch_imgs[index].cpu().numpy()*50)
    #    input()
    #index0=[index0[np.argmax(index0)]]
    #print(index0)
    #s_sum0 = lowpass_filter(s_sum0.cpu(), cutoff_freq=16, fs=1000, order=6)
    #best_gmm, bic_scores = select_gmm_components(s_sum0.cpu(), max_components=5)
    #print(best_gmm)
    #s_sum=batch_imgs.mean(dim=(1,2))
    #plt.cla()
    #plt.plot(range(s_sum0.shape[0]),s_sum0.cpu().numpy())
    #plt.plot(range(s_sum1.shape[0]),s_sum1.cpu().numpy())
    #plt.ylim(0.0,100000.0)
    #for depth in batch_imgs[200:]:
    #    cv2.imwrite("depth.png",depth.cpu().numpy()*50)
    #    print("1")
    #    input()
    #plt.savefig("plt.png")
    #input()
    #cv2.imwrite("depth.png",batch_imgs[40].cpu().numpy()*50)
    #exit()
    
    #print(closest_indices)
    #for depth in batch_imgs[closest_indices]:
    #    cv2.imwrite("depth.png",depth.cpu().numpy()*10)
    #    cv2.imwrite("depth2.png",batch_imgs[256].cpu().numpy()*10)
    #    input()   
    #input()
    #closest_indices=np.array(closest_indices)
    #base_indices=max(closest_indices)
    #for index in set(closest_indices)-set([base_indices]):
    #    print(batch_imgs[list(set(closest_indices)-set([base_indices]))].shape)
    #    exit()
    #    batch_imgs[index][(batch_imgs[index]-batch_imgs[base_indices]).abs()<0.004]=0
    return batch_imgs,closest_indices
def load_camera(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0
import tqdm
def extract_mesh(dataset, pipe, checkpoint_iterations=None):
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
    depth_index=[]
    color_list = []
    alpha_thres = 0.5
    sample_number=512
    prom=0.01
    for viewpoint_cam in tqdm.tqdm(viewpoint_cam_list[:1]):
        torch.cuda.empty_cache()
        # Rendering offscreen from that camera 
        depthes = render_threshold(viewpoint_cam, gaussians, pipe, background, kernel_size,thresholds=np.arange(0.0,0.9,0.9/sample_number))
        #cv2.imwrite(depthes)
        #depthes=kornia.filters.gaussian_blur2d(depthes.unsqueeze(1), (5,5), (1.5,1.5)).squeeze(1)
        if viewpoint_cam.gt_mask is not None:
            depthes[:,(viewpoint_cam.gt_mask < 0.5).squeeze(0)] = 0

        #for depth in depthes:
        #    cv2.imwrite("depth.png",depth.cpu().numpy()*50)
        #    input()
        #depthes=mask_peak_and_filter_batches(depthes,viewpoint_cam.gt_mask.squeeze(0))
        #print(depthes.shape)
        #for depth in depthes:
        #    cv2.imwrite("depth.png",depth.cpu().numpy()*50)
        #    input()
        #print(depthes.shape)
        depth_list.append(depthes.clone().cpu().numpy())
        #depth_index.append(index)
        del depthes
    torch.cuda.empty_cache()
    point_clouds=[]
    for depthes, viewpoint_cam in tqdm.tqdm(zip(depth_list, viewpoint_cam_list[:1])):
        W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
        fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
        fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))

        intrinsic = np.array([[fx,0,float(W)/2],[0,fy,float(H)/2],[0,0,1]],dtype=np.float64)
        intrinsic4x4 = np.eye(4, dtype=np.float64)
        intrinsic4x4[:3, :3] = intrinsic
        intrinsic = o3d.core.Tensor(intrinsic4x4)
        extrinsic = (viewpoint_cam.world_view_transform.T).cpu().numpy()

        #for depth in depth_list:
        depth=depthes[284]
        color = np.ones((H, W, 3), dtype=np.uint8) * 255  # 这里如果有颜色可替换

        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # 假设深度为米，转换到毫米

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=8.0,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W/2, H/2), extrinsic
        )
        point_clouds.append(pcd)

    # 合并所有点云
    combined_pcd = point_clouds[0]
    for pcd in point_clouds[1:]:
        combined_pcd += pcd

    # 降采样，减少点数量
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.002)

    # 保存点云
    o3d.io.write_point_cloud("recon_points.ply", combined_pcd)
    print("Point cloud generated and saved!")
    exit()
    print("done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)
        
        
    
    