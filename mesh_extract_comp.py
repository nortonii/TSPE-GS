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
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json
from gaussian_renderer import render
from tqdm import tqdm
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t
def load_camera(args, lod=0):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, lod)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import tqdm 
import trimesh
def plot_spheres(centers, radii):  
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')  

    # 设置球体的细分  
    u = np.linspace(0, 2 * np.pi, 10)  
    v = np.linspace(0, np.pi, 10)  

    # 遍历所有球体  
    for center, radius in tqdm.tqdm(zip(centers, radii)):
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))  
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))  
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))  
        ax.plot_surface(x, y, z, alpha=1.0)  # 绘制球体表面  


    plt.savefig("visAnchor.png")  
def render_depth(scene,viewpoint_cam):

    W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
    fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
    fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
    intrinsic = np.array([[fx, 0, float(W) / 2], [0, fy, float(H) / 2], [0, 0, 1]], dtype=np.float64)
    intrinsic = o3d.core.Tensor(intrinsic)
    extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))

    # 创建光线  
    rays = scene.create_rays_pinhole( 
        intrinsic,
        extrinsic,
        W,
        H
    )  

    # 进行光线投射  
    ans = scene.cast_rays(rays)  
    depth = ans['t_hit'].numpy()  # 根据光线投射结果获取深度值  
    return depth 
import glob
import open3d as o3d
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
def prepare_dataset(voxel_size,source,target):
    print(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result
def qvec2rotmat(qvec):
    # 四元数转旋转矩阵
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])

def load_colmap_poses(cameras_txt_path, images_txt_path):
    # 这里只简单读取images.txt的位姿
    pose_dict = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        elems = line.strip().split()
        if len(elems) < 10:
            continue
        image_id = int(elems[0])
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        image_name = elems[9]
        R = qvec2rotmat([qw, qx, qy, qz])
        t = np.array([tx, ty, tz]).reshape(3,1)
        # 相机到世界变换矩阵
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3, 3:] = t
        pose_dict[image_name] = pose
    return pose_dict

# 使用示例：

def extract_mesh(dataset, pipeline, checkpoint_iterations=None):
    
    with torch.no_grad():
        sample_num=1000000
        mesh_gt = o3d.io.read_triangle_mesh("/home/gyh/dataset/trans_light/boatship/boatship.ply")
        #mesh_pr = o3d.io.read_triangle_mesh(os.path.join(dataset.model_path,"recon.ply"))
        

        # 替换为你的网格文件路径  
        #mesh.compute_vertex_normals()  # 计算法线，以便光线投射更准确  
        #pose_dict = load_colmap_poses("/home/gyh/dataset/trans_light/boatship/sparse/0/cameras.txt", "/home/gyh/dataset/trans_light/boatship/sparse/0/images.txt")
        #image_paths = sorted(glob.glob(os.path.join(dataset.source_path+"/images", "*.png")))
        #image_name0 = image_paths[0].split('/')[-1]  # 获取第0张图的文件名
        #pose0 = pose_dict[image_name0]  # 第0张图的世界变换矩阵

        # 变换网格，照你原来的逻辑相机位姿取逆：
        #mesh_gt.transform(np.linalg.inv(pose0))

        # 创建Raycasting场景  
        scene = o3d.t.geometry.RaycastingScene()  
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_gt))  
        """
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                                  dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                                  dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                                  dataset.add_color_dist)
        gaussians.eval()
        
        output_path = os.path.join(dataset.model_path, "point_cloud")
        iteration = 0
        if checkpoint_iterations is None:
            for folder_name in os.listdir(output_path):
                iteration = max(iteration, int(folder_name.split('_')[1]))
        else:
            iteration = checkpoint_iterations

        output_ply_path = os.path.join(output_path, "iteration_" + str(iteration), "point_cloud.ply")
        output_mlp_path = os.path.join(output_path, "iteration_" + str(iteration))
        gaussians.load_ply_sparse_gaussian(output_ply_path)
        gaussians.load_mlp_checkpoints(output_mlp_path)
        """
        #points=gaussians.get_scaling[:,:3].unsqueeze(1)*gaussians._offset
        #scales=gaussians._scaling.detach().cpu().numpy()
        #centroid = torch.mean(points, dim=1,keepdim=True)
        #distances = torch.norm(points - centroid, dim=2)  
        #radius = torch.max(distances,dim=1)[0]
        #print(torch.norm(gaussians._anchor,dim=1).shape)
        #plt.hist(torch.norm(gaussians._anchor,dim=1).cpu(),100)
        #plt.savefig("test.png")
        #exit()
        
        #print(f'Loaded gaussians from {output_path}')

        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewpoint_cam_list = load_camera(dataset)

        depth_list = []
        edepth_list = []
        color_list = []
        normal_list=[]
        alpha_thres = 0.5
        import cv2
        for viewpoint_cam in viewpoint_cam_list:
            # Rendering offscreen from that camera
            """
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipeline, background)
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
            rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1, 2, 0)
            depth = render_pkg["median_depth"].clone()
            edepth = render_pkg["expected_depth"].clone()
            normal = render_pkg["normal"].clone()
            if viewpoint_cam.mask is not None:
                depth[(viewpoint_cam.mask < 0.5)] = 0
            color_list.append(np.ascontiguousarray(rendered_img))
            #depth[render_pkg["mask"] < alpha_thres] = 0
            edepth[render_pkg["mask"] < alpha_thres] = 0
            depth_list.append(depth[0].cpu().numpy())
            edepth_list.append(np.abs(edepth[0].cpu().numpy()-depth[0].cpu().numpy()))


            import cv2
            plt.cla()
            """
            depth_gt=render_depth(scene=scene,viewpoint_cam=viewpoint_cam)
            cv2.imwrite('depth.png', depth_gt*50.0)
            input()
            continue
            diff = depth.cpu().detach().squeeze(0).numpy() - depth_gt  
            
            # 创建红色和蓝色通道  
            error_red = (diff <-0.3)*255#np.abs(diff)*255  # 负误差  
            error_blue = (diff > 0.3)*255#*np.abs(diff)*255  # 正误差  
            print((diff <-0.3).sum())
            print((diff > 0.3).sum())
            # 创建三通道图像 (BGR格式)  
            error_image = render_pkg["render"].cpu().detach().numpy().transpose(1,2,0)*100+np.stack((error_blue, np.zeros_like(error_red), error_red), axis=2).astype(np.uint8)  

            # 保存图像  
            cv2.imwrite('color_gradient.png', error_image)
            cv2.imwrite('depth.png', depth.cpu().detach().squeeze(0).numpy()*20)
            cv2.imwrite('gtdepth.png', depth_gt*20)
            input()
            normal_list.append(normal)
        exit()
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
        import tqdm
        import time
        trunc = voxel_size * 4
        res = 8
        device=o3d_device
        
        for color, depth,erord,viewpoint_cam in tqdm.tqdm(zip(color_list, depth_list,edepth_list, viewpoint_cam_list)):
            depth = o3d.t.geometry.Image(depth)
            depth = depth.to(o3d_device)
            color = o3d.t.geometry.Image(color)
            color = color.to(o3d_device)
            #P=(viewpoint_cam.projection_matrix).cuda()
            #P_inv_T = torch.inverse(P).t()
            #ones = torch.ones((1,normals.shape[1],normals.shape[2])).cuda()
            #normals_4d = torch.cat((normals.cuda(), ones))
            #ray_normal=torch.einsum('ij,jkt->ikt',P_inv_T,normals_4d)
            #ray_normal=torch.nn.functional.normalize((ray_normal[:3,...]),dim=0)
            #theta = torch.acos(ray_normal[2])
            #theta =torch.clamp(theta,0.5,1.0)

            W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
            fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
            fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
            intrinsic = np.array([[fx, 0, float(W) / 2], [0, fy, float(H) / 2], [0, 0, 1]], dtype=np.float64)
            intrinsic = o3d.core.Tensor(intrinsic)
            extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
            config_depth_scale=1.0
            config_depth_max=8.0
            start = time.time()
            # Get active frustum block coordinates from input
            #frustum_block_coords = vbg.compute_unique_block_coordinates(
            #    depth, intrinsic, extrinsic, config_depth_scale, config_depth_max)
            print(type(o3d.utility.Vector3dVector(render_pkg["xyz"].cpu())))
            pcd=o3d.cuda.pybind.t.geometry.PointCloud(o3d.core.Tensor(render_pkg["xyz"].cpu().numpy(), dtype=o3d.core.Dtype.Float32, device=device))
            #pcd=pcd.cpu()
            #vbg=vbg.cpu()
            frustum_block_coords = vbg.compute_unique_block_coordinates(pcd)
            vbg.hashmap().activate(frustum_block_coords)
            buf_indices, masks = vbg.hashmap().find(frustum_block_coords)
            tsdf = vbg.attribute('tsdf').reshape((-1, 1))
            voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(
                buf_indices)
            print(tsdf[voxel_indices].sum())
            
            exit()
            # Activate them in the underlying hash map (may have been inserted)
            vbg.hashmap().activate(frustum_block_coords)

            # Find buf indices in the underlying engine
            buf_indices, masks = vbg.hashmap().find(frustum_block_coords)
            end = time.time()

            start = time.time()
            voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(
                buf_indices)
            end = time.time()

            # Now project them to the depth and find association
            # (3, N) -> (2, N)
            start = time.time()
            extrinsic_dev = extrinsic.to(device, o3c.float32)
            xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3, 3:]

            intrinsic_dev = intrinsic.to(device, o3c.float32)
            uvd = intrinsic_dev @ xyz
            d = uvd[2]
            u = (uvd[0] / d).round().to(o3c.int64)
            v = (uvd[1] / d).round().to(o3c.int64)
            o3d.core.cuda.synchronize()
            end = time.time()

            start = time.time()
            mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < depth.columns) & (
                v < depth.rows)

            v_proj = v[mask_proj]
            u_proj = u[mask_proj]
            d_proj = d[mask_proj]
            depth_readings = depth.as_tensor()[v_proj, u_proj, 0].to(
                o3c.float32) / config_depth_scale
            sdf = depth_readings - d_proj

            mask_inlier = (depth_readings > 0) \
                & (depth_readings < config_depth_max) \
                & (sdf >= -trunc)

            sdf[sdf >= trunc] = trunc
            sdf = sdf / trunc
            end = time.time()

            start = time.time()
            weight = vbg.attribute('weight').reshape((-1, 1))
            tsdf = vbg.attribute('tsdf').reshape((-1, 1))

            valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
            w = weight[valid_voxel_indices]
            error_weight=torch.clamp(torch.tensor(1.0-erord),0.0,1.0).cuda()
            error_weight = error_weight[v_proj.numpy(), u_proj.numpy()]
            error_weight=(error_weight[mask_inlier.numpy()]).unsqueeze(1).cpu().numpy()
            wp = w + error_weight#theta_weight

            tsdf[valid_voxel_indices] \
                = (tsdf[valid_voxel_indices] * error_weight +
                sdf[mask_inlier].reshape(w.shape)) / (wp)
            """
            if config.integrate_color:
                color = o3d.t.io.read_image(color_file_names[i]).to(device)
                color_readings = color.as_tensor()[v_proj, u_proj].to(o3c.float32)

                color = vbg.attribute('color').reshape((-1, 3))
                color[valid_voxel_indices] \
                    = (color[valid_voxel_indices] * w +
                            color_readings[mask_inlier]) / (wp)
            """
            weight[valid_voxel_indices] = wp

        mesh = vbg.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, "recon.ply"), mesh.to_legacy())
        print("done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)