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
os.environ['CUDA_VISIBLE_DEVICES']=str(1)

os.system('echo $CUDA_VISIBLE_DEVICES')

def kmeans_pytorch(X, k, num_iters=100):
    N, D = X.shape
    indices = torch.randperm(N)[:k]
    centroids = X[indices]

    for _ in range(num_iters):
        dists = torch.cdist(X, centroids, p=2)
        cluster_ids = torch.argmin(dists, dim=1)
        new_centroids = torch.stack([X[cluster_ids == i].mean(dim=0) if (cluster_ids == i).any() else centroids[i] for i in range(k)])
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    return cluster_ids, centroids

def compute_sse(X, cluster_ids, centroids):
    sse = 0.0
    for i in range(centroids.shape[0]):
        cluster_points = X[cluster_ids == i]
        if cluster_points.shape[0] > 0:
            distances = torch.norm(cluster_points - centroids[i], dim=1)
            sse += (distances ** 2).sum().item()
    return sse

def compute_sse_for_k_range(X, k_range, num_iters=100):
    sse_dict = {}
    for k in k_range:
        cluster_ids, centroids = kmeans_pytorch(X, k, num_iters)
        sse = compute_sse(X, cluster_ids, centroids)
        sse_dict[k] = sse
        print(f"k={k}, SSE={sse:.4f}")
    return sse_dict

def find_elbow_k(sse_dict):
    """
    Simple elbow detection: 
    Here we choose the k where the decrease in SSE slows down significantly.
    More sophisticated automatic elbow detection can be implemented if needed.
    """
    ks = sorted(sse_dict.keys())
    sse_values = [sse_dict[k] for k in ks]
    deltas = np.diff(sse_values)
    second_deltas = np.diff(deltas)
    # Find k corresponding to max second difference negative peak (max curvature)
    if len(second_deltas) == 0:
        return ks[0]
    elbow_index = np.argmin(second_deltas) + 2  # +2 because diff reduces len by 1 twice
    return ks[elbow_index]

def run_clustering_pipeline(X, cluster_centers=None, std=1.5, k_min=1, k_max=10, num_iters=100):
    k_range = range(k_min, k_max + 1)
    sse_dict = compute_sse_for_k_range(X, k_range, num_iters)
    best_k = find_elbow_k(sse_dict)
    print(f"Selected best cluster number by elbow method: {best_k}")
    final_cluster_ids, final_centroids = kmeans_pytorch(X, best_k, num_iters)
    print(f"Final centroids:\n{final_centroids.squeeze().numpy()}")
    return best_k, final_cluster_ids, final_centroids, sse_dict
def lowpass_filter(data_tensor, cutoff_freq, fs, order=4):
    """
    参数：
        data_tensor: 1D torch.Tensor信号
        cutoff_freq: 截止频率（Hz）
        fs: 采样频率（Hz）
        order: 滤波器阶数，越高越陡峭
    返回：
        filtered_data: 低通滤波后的信号（torch.Tensor）
    """
    data_np = data_tensor.numpy()
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data_np)
    print(filtered.shape)
    return torch.tensor(filtered.copy())
def custom_bic(gmm: GaussianMixture, X: np.ndarray, weight_lambda=1000000000.0):
    """
    自定义BIC计算，允许调节复杂度惩罚权重
    参数:
        gmm: 训练好的GaussianMixture模型
        X: 训练数据，形状(n_samples, n_features)
        weight_lambda: 惩罚权重系数，默认1.0对应sklearn默认的BIC
    返回:
        自定义权重后的BIC值
    """
    n_samples, n_features = X.shape
    log_likelihood = gmm.score(X) * n_samples  # 总对数似然（score返回均值对数似然）
    n_params = gmm._n_parameters()
    bic = -2 * log_likelihood + n_params ** np.log(n_samples)
    return bic

def select_gmm_components(data_tensor, max_components=10, weight_lambda=1.0):
    data_np = data_tensor.numpy().reshape(-1, 1)
    lowest_bic = np.inf
    best_gmm = None
    bic_scores = []

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        gmm.fit(data_np)
        bic = custom_bic(gmm, data_np, weight_lambda=weight_lambda)
        bic_scores.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    return best_gmm, bic_scores

URL = "http://localhost:5000/cluster_images?n_clusters=2"

def upload_files_multipart(filepaths):
    """
    方式1: 上传本地图片文件 多文件form-data
    """
    files = [('images', (open(fp, 'rb'))) for fp in filepaths]
    response = requests.post(URL, files=files)
    print("Multipart upload response:", response.json())

def upload_base64_color(images_pil):
    """
    方式2: 上传彩色PIL图list，base64 JSON数组
    """
    b64_list = []
    for img in images_pil:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        b64_str = base64.b64encode(buffered.getvalue()).decode()
        b64_list.append(b64_str)
    json_data = {"images": b64_list}
    response = requests.post(URL, json=json_data)
    print("Base64 color upload response:", response.json())

def upload_base64_gray(np_imgs):
    """
    方式3: 上传灰度np.ndarray (N,400,400), numpy uint8数组 base64 JSON数组
    每张图转bytes再base64编码
    """
    b64_list = []
    for i in range(np_imgs.shape[0]):
        img_bytes = np_imgs[i].astype(np.uint8).tobytes()
        b64_str = base64.b64encode(img_bytes).decode()
        b64_list.append(b64_str)
    json_data = {"gray_images": b64_list}
    response = requests.post(URL, json=json_data)
    return response.json()

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
def compute_bics(X, k_min=1, k_max=10):
    bics = []
    ks = list(range(k_min, k_max+1))
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        bics.append(bic)
        print(f"k={k}, BIC={bic:.2f}")
    return ks, np.array(bics)

def find_elbow_k_bic(ks, bics):
    bics=torch.tensor(bics)
    print(bics[:-1])
    print(torch.diff(bics))
    if (torch.diff(bics)>(0.2*bics[:-1])).sum()==0:
        return 1
    best_k=torch.where(torch.diff(bics)>(0.2*bics[:-1]))[-1].item() + 2  # 找到第一个BIC下降超过阈值的k
    return best_k
def fit_gmm_and_select_k(X, k_min=1, k_max=5):
    bics = []
    ks = range(k_min, k_max+1)

    best_gmm = None
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(X)
        #bic = gmm.bic(X)
        bic = gmm.score(X)
        bics.append(bic)
        print(f"k={k}, BIC={bic:.2f}")

    best_k = find_elbow_k_bic(ks, bics)
    print(f"Selected best k by BIC: {best_k}")
    

    # 训练最终的最佳模型
    best_gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
    best_gmm.fit(X)
    print(f"Mix weight: {best_gmm.weights_}")

    return best_k, bics, best_gmm

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

def remove_adjacent_duplicates(batch_imgs: torch.Tensor, gt_mask,prom):

    s_sum1= batch_imgs.mean(dim=(1, 2))  # 计算每张图的平均值

    data_np= s_sum1.cpu().numpy()
    kde = gaussian_kde(data_np)
    #plt.cla()
    #fig = plt.figure(dpi=300, facecolor="#EFE9E6")
    #ax = plt.subplot(111, facecolor="#EFE9E6")

    value_range = data_np.max() - data_np.min()
    x_eval = np.linspace(data_np.min() - value_range * 0.02, data_np.max(), 1000)
    density = kde(x_eval)

    density = density / density.sum()

    #ax.plot(np.array(list(range(1000)))/1000.0, density, linewidth=3, color="#ffa863", linestyle='solid', label="Density")

    #ax.spines["left"].set_visible(False)
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)
    #ax.grid(ls="--", lw=0.5, color="#4E616C")

    peaks, pin = find_peaks(density, prominence=prom * (density.max() - density.min()))
    peak_positions = x_eval[peaks]

    closest_indices = []
    for pos in peak_positions:
        idx = np.argmin(np.abs(data_np - pos))
        closest_indices.append(idx)
    #print(closest_indices)

    #ax.set_xlabel("Transmission", fontsize=12)
    #ax.set_ylabel("Density", fontsize=12)
    #ax.legend()

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
    #    cv2.imwrite("depth.png",depth.cpu().numpy()*50)
    #    input()   
    #input()
    #closest_indices=np.array(closest_indices)
    #base_indices=max(closest_indices)
    #for index in set(closest_indices)-set([base_indices]):
    #    print(batch_imgs[list(set(closest_indices)-set([base_indices]))].shape)
    #    exit()
    #    batch_imgs[index][(batch_imgs[index]-batch_imgs[base_indices]).abs()<0.004]=0
    return batch_imgs[closest_indices],closest_indices
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
    import time

    start = time.perf_counter()  # 高精度计时

    for viewpoint_cam in tqdm.tqdm(viewpoint_cam_list):
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
        depthes,index=remove_adjacent_duplicates(depthes,viewpoint_cam.gt_mask.squeeze(0),prom)
        #for depth in depthes:
        #    cv2.imwrite("depth.png",depth.cpu().numpy()*50)
        #    input()
        #print(depthes.shape)
        depth_list.append(depthes.clone().cpu().numpy())
        depth_index.append(index)
        del depthes,index
    torch.cuda.empty_cache()
    single=True
    merged_list=[]
    for sublist in depth_index:
        merged_list.extend([item  for item in sublist])
        if len(sublist)>=2:
            single=False
    if not single:
        data = np.array(merged_list).reshape(-1, 1)  # KMeans 需要二维数组

        # 进行 KMeans 聚类
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(data)
        print(kmeans.labels_)
        clustered = {}
        for label, value in zip(kmeans.labels_, merged_list):
            clustered[value]=label
        depth_label=[]
        depth_in=[]
        depth_out=[]
        for depthes,index in zip(depth_list,depth_index):
            temp_in=[]
            temp_out=[]
            for i,item in enumerate(index):
                if clustered[item]==1:
                    temp_out.append(i)
                else:
                    temp_in.append(i)
            depth_in.append(depthes[temp_in])
            depth_out.append(depthes[temp_out])
    else:
        depth_out=[]
        for depthes,index in zip(depth_list,depth_index):
            depth_out.append(depthes[index])
        depth_in=[]
    voxel_size = 0.002
    o3d_device = o3d.core.Device("CPU:0")

    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
                                            attr_dtypes=(o3c.float32,
                                                         o3c.float32),
                                            attr_channels=((1), (1)),
                                            voxel_size=voxel_size,
                                            block_resolution=16,
                                            block_count=50000,
                                            device=o3d_device)
    impacted_blocks=set()
    for depthes, viewpoint_cam in tqdm.tqdm(zip(depth_out, viewpoint_cam_list)):
        for depth in depthes:
            #cv2.imwrite("depth.png",depth*50)
            #input()
            depth = o3d.t.geometry.Image(depth)
            depth = depth.to(o3d_device)
            W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
            fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
            fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
            intrinsic = np.array([[fx,0,float(W)/2],[0,fy,float(H)/2],[0,0,1]],dtype=np.float64)
            intrinsic = o3d.core.Tensor(intrinsic)
            extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
            occu_coords = vbg.compute_unique_block_coordinates(
                                                                            depth, 
                                                                            intrinsic,
                                                                            extrinsic, 
                                                                            1.0, 8.0,
                                                                            trunc_voxel_multiplier=0.25
                                                                        )
            coords_np = occu_coords.cpu().numpy()
            for coord in coords_np:
                impacted_blocks.add(tuple(coord))
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                                                                            depth, 
                                                                            intrinsic,
                                                                            extrinsic, 
                                                                            1.0, 8.0
                                                                        )
            vbg.integrate(
                            frustum_block_coords, 
                            depth, 
                            intrinsic,
                            extrinsic,  
                            1.0, 8.0
                        )

    mesh = vbg.extract_triangle_mesh()
    #mesh = post_process_mesh(mesh.to_legacy(),cluster_to_keep=1)
    #mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path,"recon_opa.ply"),mesh.to_legacy())
    flag=True
    for depthes, viewpoint_cam in tqdm.tqdm(zip(depth_in, viewpoint_cam_list)):
        for depth in depthes:
            if depth.shape[0]!=0:
                flag=False
                break
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path,"recon_all"+str(prom)+".ply"),mesh.to_legacy())
    if flag:
        print("done")
        exit()
    del vbg
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
                                            attr_dtypes=(o3c.float32,
                                                         o3c.float32),
                                            attr_channels=((1), (1)),
                                            voxel_size=voxel_size,
                                            block_resolution=16,
                                            block_count=50000,
                                            device=o3d_device)
    for depthes, viewpoint_cam in tqdm.tqdm(zip(depth_in, viewpoint_cam_list)):
        for depth in depthes:
            depth = o3d.t.geometry.Image(depth)
            depth = depth.to(o3d_device)
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
            coords_np = frustum_block_coords.cpu().numpy()
            filtered_coords = []
            for coord in coords_np:
                if tuple(coord) not in impacted_blocks:
                    filtered_coords.append(coord)
            if len(filtered_coords) == 0:
                continue  # 全是已存在的Block，跳过
            
            filtered_coords_tensor = o3d.core.Tensor(np.array(filtered_coords, dtype=np.int32))
            vbg.integrate(
                            filtered_coords_tensor, 
                            depth, 
                            intrinsic,
                            extrinsic,  
                            1.0, 8.0
                        )

    mesh = vbg.extract_triangle_mesh()
    #o3d.io.write_triangle_mesh(os.path.join(dataset.model_path,"recon_opa_inner.ply"),mesh.to_legacy())
    mesh_outer=o3d.io.read_triangle_mesh(os.path.join(dataset.model_path,"recon_opa.ply"))
    mesh_inner=mesh.to_legacy()
    merge_mesh=mesh_outer+mesh_inner
    merge_mesh.remove_duplicated_vertices()
    merge_mesh.remove_duplicated_triangles()
    merge_mesh.remove_degenerate_triangles()
 
    # 可选：合并近距离的顶点
    # merged_mesh.merge_close_vertices(distance=0.001)
 
    # 保存或可视化合并后的多边形模型
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path,"recon_all"+str(prom)+".ply"),merge_mesh)
    print("done!")
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)
        
        
    
    