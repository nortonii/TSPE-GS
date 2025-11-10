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
from gaussian_renderer import render,render_opa
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
def remove_adjacent_duplicates(batch_imgs: torch.Tensor, similarity_threshold=0.98):
    labels=upload_base64_gray(batch_imgs.cpu().numpy())["center_indices"]
    #for label in labels:
    #    cv2.imwrite("depth.png",batch_imgs[label].cpu().numpy()*50)
    #    input()

    return batch_imgs[labels]
def load_camera(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
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
    color_list = []
    alpha_thres = 0.5
    
    for viewpoint_cam in viewpoint_cam_list:
        # Rendering offscreen from that camera 
        depthes = render_opa(viewpoint_cam, gaussians, pipe, background, kernel_size,opa_multis=np.arange(0.01,0.01*128,0.01))
        #cv2.imwrite(depthes)
        #depthes=kornia.filters.gaussian_blur2d(depthes.unsqueeze(1), (5,5), (1.5,1.5)).squeeze(1)
        if viewpoint_cam.gt_mask is not None:
            depthes[:,(viewpoint_cam.gt_mask < 0.5).squeeze(0)] = 0
        #for depth in depthes:
        #    cv2.imwrite("depth.png",depth.cpu().numpy()*50)
        #    input()
        depthes=remove_adjacent_duplicates(depthes)
        #for depth in depthes:
        #    cv2.imwrite("depth.png",depth.cpu().numpy()*50)
        #    input()
        #print(depthes.shape)
        depth_list.append(depthes.clone().cpu().numpy())
        del depthes
    torch.cuda.empty_cache()
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
    import tqdm
    for depthes, viewpoint_cam in tqdm.tqdm(zip(depth_list, viewpoint_cam_list)):
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
            vbg.integrate(
                            frustum_block_coords, 
                            depth, 
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
        
        
    
    