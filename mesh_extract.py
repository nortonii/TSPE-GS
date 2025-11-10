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
from gaussian_renderer import render
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json


def load_camera(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
def poisson_reconstruct_from_tensor(tensor, depth=9, width=0, scale=1.1, linear_fit=False,
                                     estimate_normals_knn=30, output_mesh_path="poisson_mesh.ply"):
    """
    Args:
        tensor: PyTorch tensor of shape (N,3), device can be cuda; dtype float32/64.
        depth: int, Octree depth parameter for Poisson (higher = more detailed, memory increases).
        width: int, 0 (default) lets Open3D choose.
        scale: float, bounding box scale for reconstruction.
        linear_fit: bool, whether to use linear interpolation in Poisson.
        estimate_normals_knn: int, neighbors used to estimate normals if not present.
        output_mesh_path: str, path to save the reconstructed mesh (.ply).
    Returns:
        mesh: open3d.geometry.TriangleMesh object (reconstructed mesh).
    """

    # 1) Convert to numpy
    if tensor.requires_grad:
        tensor = tensor.detach()
    cpu_tensor = tensor.to('cpu').float()
    pts = cpu_tensor.numpy()
    assert pts.ndim == 2 and pts.shape[1] == 3, "Input tensor must be (N,3)"

    # 2) Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 3) Estimate normals if not present
    #    Open3D can estimate normals via KDTree search. Orientation may be inconsistent; we can orient by PCA or camera.
    print("Estimating normals with knn =", estimate_normals_knn)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=estimate_normals_knn))
    # Optionally, orient normals consistently (towards camera / view point)
    # Choose a viewpoint outside the point cloud (e.g., mean + z offset)
    try:
        center = pcd.get_center()
        viewpoint = center + np.array([0.0, 0.0, 1.0])  # adjust if needed
        pcd.orient_normals_towards_camera_location(viewpoint)
    except Exception as e:
        print("Warning: orient_normals_towards_camera_location failed:", e)

    # 4) Poisson reconstruction
    print(f"Running Poisson reconstruction (depth={depth}, scale={scale}, linear_fit={linear_fit}) ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )

    # 5) Optional: Crop low-density vertices (remove spurious geometry)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)  # remove lowest 1% density vertices
    vertices_to_keep = densities > density_threshold
    # Create vertex mask
    import copy
    mesh_clean = mesh.select_by_index(np.where(vertices_to_keep)[0])
    # Another strategy: bounding box crop to original points
    # bbox = pcd.get_axis_aligned_bounding_box()
    # mesh_clean = mesh.crop(bbox)

    # 6) Save mesh
    o3d.io.write_triangle_mesh(output_mesh_path, mesh_clean)
    print("Saved mesh to", output_mesh_path)

    return mesh_clean



def extract_mesh(dataset, pipe, checkpoint_iterations=None):
    import time

    start = time.perf_counter()  # 高精度计时
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
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1,2,0)
        color_list.append(np.ascontiguousarray(rendered_img))
        depth = render_pkg["median_depth"].clone()

        #import cv2
        #print(depth.shape)
        #cv2.imwrite("image.png",rendered_img*255)
        #input()
        if viewpoint_cam.gt_mask is not None:
            depth[(viewpoint_cam.gt_mask < 0.5)] = 0
        depth[render_pkg["mask"]<alpha_thres] = 0
        depth_list.append(depth[0].cpu().numpy())

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
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path,"recon_old.ply"),mesh.to_legacy())
    print("done!")
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")
    exit()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)
        
        
    
    