# Script modified from https://github.com/jzhangbs/DTUeval-python

from genericpath import isdir
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
import torch

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1].astype(np.float32)
    c += 0.5
    denom = max(n1, 1e-7)
    c[0] /= denom
    denom = max(n2, 1e-7)
    c[1] /= denom
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def sample_triangles_gpu_fully_batched(n1_batch, n2_batch, v1_batch, v2_batch, tri_vert_batch):
    """
    输入：
        n1_batch, n2_batch: 长度B的LongTensor，指示每个三角形的采样密度
        v1_batch, v2_batch, tri_vert_batch: float tensor (B,3)
    输出：
        所有三角形采样点合并后的Tensor (N,3)，按GPU张量返回
    """

    device = v1_batch.device
    B = len(n1_batch)

    max_n1 = n1_batch.max().item()
    max_n2 = n2_batch.max().item()

    # 先生成固定最大网格坐标 (max_n1+1, max_n2+1, 2)
    c_x = (torch.arange(max_n1 + 1, device=device).float() + 0.5) / max(1, max_n1)
    c_y = (torch.arange(max_n2 + 1, device=device).float() + 0.5) / max(1, max_n2)
    grid_x, grid_y = torch.meshgrid(c_x, c_y, indexing='ij')  # shapes (max_n1+1, max_n2+1)

    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (M,2), M = (max_n1+1)*(max_n2+1)

    # 三角形级别的mask，判断点是否满足 x + y < 1（对所有点相同）
    mask = (grid[:, 0] + grid[:, 1]) < 1  # (M,)
    masked_grid = grid[mask]  # (M2,2), M2 < M

    M2 = masked_grid.shape[0]

    # 对每个三角形，采取前n1+1, n2+1的采样网格，裁剪掩码
    # 构造每个三角形的掩码（M2,)，表示哪些mask点落在其采样范围内
    # 方式：用真实坐标 * n1和 n2 判断是否 <= n1,n2

    # 扩展维度方便广播计算 (B, M2, 2)
    grid_exp = masked_grid.unsqueeze(0).repeat(B, 1, 1)  # (B, M2, 2)

    n1_exp = n1_batch.unsqueeze(1).float()  # (B, 1)
    n2_exp = n2_batch.unsqueeze(1).float()  # (B, 1)

    # 对每个点判断是否落入各三角形采样网格范围内
    inside_mask = (grid_exp[:, :, 0] * max_n1 <= n1_exp) & (grid_exp[:, :, 1] * max_n2 <= n2_exp)  # (B, M2)

    # 用 inside_mask 给 grid_exp 做过滤
    # 由于每个三角形采样点数不同，后面需要把不同点拼接，所以分配索引后再拼接。

    output_pts = []

    for i in range(B):
        pts_in = grid_exp[i][inside_mask[i]].unsqueeze(0)  # (1, n_pts_i, 2)
        # 按比例计算三角形对应的 (x,y) 坐标（采样比例）
        scale_x = n1_batch[i].float()
        scale_y = n2_batch[i].float()

        if pts_in.shape[1] == 0:
            continue

        # 获取v1, v2, tri_vert
        v1 = v1_batch[i:i+1]  # (1,3)
        v2 = v2_batch[i:i+1]  # (1,3)
        tri_vert = tri_vert_batch[i:i+1]  # (1,3)

        # 缩放采样点由[0,1]变到实际网格比例
        pts_scaled = pts_in * torch.tensor([scale_x / max_n1, scale_y / max_n2], device=device)

        # 采样点实际位置
        pts_3d = v1 * pts_scaled[:, :, 0:1] + v2 * pts_scaled[:, :, 1:2] + tri_vert  # (1, n_pts_i, 3)

        output_pts.append(pts_3d.reshape(-1, 3))  # (n_pts_i, 3)

    if len(output_pts) == 0:
        return torch.empty((0, 3), device=device)

    return torch.cat(output_pts, dim=0)  # (N,3)

def sample_mesh(data_mesh, nop):
    return np.asarray(data_mesh.sample_points_uniformly(number_of_points=nop).points)


from scipy.spatial import cKDTree
import numpy as np
import tqdm
import gc
from sklearn.neighbors import BallTree
import tqdm
import gc

def eval_cf_fast_chunked_balltree(pred, gt, chunk_size=10000):
    tree_gt = BallTree(gt)
    tree_pred = BallTree(pred)
    print("build tree done")

    dist_d2s_list = []
    for i in tqdm.tqdm(range(0, len(pred), chunk_size)):
        chunk_pred = pred[i:i+chunk_size]
        # BallTree query 返回距离和索引，k=1最近邻
        dist_chunk, _ = tree_gt.query(chunk_pred, k=1)
        dist_d2s_list.append(dist_chunk.flatten())  # flatten成1D数组
        del chunk_pred, dist_chunk
        gc.collect()

    dist_d2s = np.concatenate(dist_d2s_list)
    del dist_d2s_list
    gc.collect()

    dist_s2d_list = []
    for j in tqdm.tqdm(range(0, len(gt), chunk_size)):
        chunk_gt = gt[j:j+chunk_size]
        dist_chunk, _ = tree_pred.query(chunk_gt, k=1)
        dist_s2d_list.append(dist_chunk.flatten())
        del chunk_gt, dist_chunk
        gc.collect()

    dist_s2d = np.concatenate(dist_s2d_list)
    del dist_s2d_list
    gc.collect()

    return dist_d2s, dist_s2d


if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='',
                        help='path to predicted pts')
    parser.add_argument('--gt_path', default=None,
                        help='path to extracted ground turth pts')

    parser.add_argument('--downsample_density', type=float, default=0.001)
    parser.add_argument('--pt_downsample', action='store_true', default=False)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=1.5)
    parser.add_argument('--f1_dist', type=float, default=0.01,
                        help="the distance threshold for ac1uracy/completeness/f1 metrics")
    parser.add_argument('--visualize_threshold', type=float, default=0.1)
    parser.add_argument('--run_alpha_shape', action='store_true', default=False,
                        help='convert point to mesh, then eval cf')
    parser.add_argument('--alpha_shape_alpha', type=float, default=0.003)
    parser.add_argument('--out_dir', type=str, default='./')
    # parser.add_argument('--del_ckpt', action='store_true', default=False)
    parser.add_argument('--no_pts_save', action='store_true', default=False)
    parser.add_argument('--log_tune_hparam_config_path', type=str, default=None,
                        help='Log hyperparamters being tuned to tensorboard based on givn config.json path')
    parser.add_argument('--hparam_save_name', type=str, default='hparam_pts')
    args = parser.parse_args()
    nop = 5000000
    thresh = args.downsample_density
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    summary_writer = SummaryWriter(f'{os.path.dirname(args.input_path)}/../')

    if args.gt_path is not None:
        if os.path.isdir(args.gt_path):
            stl = np.load(f'{args.gt_path}/shape.npy')
        else:
            stl = np.load(args.gt_path)
        mesh_eval = False

    if args.input_path.endswith('.obj') or args.input_path.endswith('.ply'):
        mesh_eval = True
        # read from mesh
        data_mesh = o3d.io.read_triangle_mesh(args.input_path)
        data_pcd = sample_mesh(data_mesh, nop)

    else:
        if os.path.isdir(args.input_path):
            data_pcd = np.load(f'{args.input_path}/pts.npy')
        else:
            data_pcd = np.load(args.input_path)

        if args.pt_downsample:
            nn_engine.fit(data_pcd)
            rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
            mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
            for curr, idxs in enumerate(rnn_idxs):
                if mask[curr]:
                    mask[idxs] = 0
                    mask[curr] = 1
            data_pcd = data_pcd[mask]

        if args.run_alpha_shape:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data_pcd)
            print('running alpha shape')
            data_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, args.alpha_shape_alpha)
            print('alpha shape finished')

            os.makedirs(args.out_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(f'{args.out_dir}/mesh_{args.alpha_shape_alpha}.ply', data_mesh)

            data_pcd = sample_mesh(data_mesh, nop)

    if args.gt_path is None:
        # save points only
        os.makedirs(args.out_dir, exist_ok=True)
        write_vis_pcd(f'{args.out_dir}/vis_d2s.ply', data_pcd)
        exit(0)
    print(4)
    dist_d2s, dist_s2d = eval_cf_fast_chunked_balltree(data_pcd, stl)
    max_dist = args.max_dist
    print(5)
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    #accuracy = np.count_nonzero(dist_d2s < args.f1_dist) / len(dist_d2s)
    #completeness = np.count_nonzero(dist_s2d < args.f1_dist) / len(dist_s2d)
    #f1 = 2. / (1. / np.clip(accuracy, 1e-10, None) + 1. / np.clip(completeness, 1e-10, None))
    print(6)
    vis_dist = args.visualize_threshold
    R = np.array([[1, 0, 0]], dtype=np.float64)
    G = np.array([[0, 1, 0]], dtype=np.float64)
    B = np.array([[0, 0, 1]], dtype=np.float64)
    W = np.array([[1, 1, 1]], dtype=np.float64)
    data_pcd = data_pcd[dist_d2s < max_dist]  # remove redundent points
    dist_d2s = dist_d2s[dist_d2s < max_dist]  # remove redundent points
    #data_color = np.tile(B, (data_pcd.shape[0], 1))
    #data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    #data_color = R * data_alpha + W * (1 - data_alpha)
    #data_color[dist_d2s[:, 0] >= max_dist] = G
    #stl_color = np.tile(B, (stl.shape[0], 1))
    #stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    #stl_color = R * stl_alpha + W * (1 - stl_alpha)
    #stl_color[dist_s2d[:, 0] >= max_dist] = G
    over_all = (mean_d2s + mean_s2d) / 2
    print(7)
    # read image eval metrics if avaliable
    img_eval_path = Path(args.input_path).parent / '..' / 'render_eval.txt'
    if not img_eval_path.exists():
        img_eval_path = Path(args.input_path).parent / '..' / '..' / 'render_eval.txt'

    if img_eval_path.exists():
        with img_eval_path.open('r') as f:
            psnr = float(f.readline().split(':')[-1].strip())
    else:
        psnr = None

    print(f'======= eval result =======')
    print(f'Mean d2s: {mean_d2s}')
    print(f'Mean s2d: {mean_s2d}')
    print(f'Avg cf: {over_all}')
    exit()
    #print(f'Accuracy: {accuracy}')
    #print(f'Completeness: {completeness}')
    #print(f'F1: {f1}')
    #print(f'psnr: {psnr}\n')
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        if not args.no_pts_save:
            write_vis_pcd(f'{args.out_dir}/vis_d2s.ply', data_pcd, data_color)
            write_vis_pcd(f'{args.out_dir}/vis_s2d.ply', stl, stl_color)

        with open(f'{args.out_dir}/cf.txt', 'w') as f:
            f.write(f'Mean d2s: {mean_d2s}\n')
            f.write(f'Mean s2d: {mean_s2d}\n')
            f.write(f'Avg cf: {over_all}\n')

            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Completeness: {completeness}\n')
            f.write(f'F1: {f1}\n')

            f.write(f'psnr: {psnr}\n')

        print(f'Result output to {args.out_dir}/cf.txt')

        if psnr is not None:
            metrics['Image/psnr'] = psnr

        # summary_writer.add_hparams(hparams, metrics, run_name=os.path.realpath(f'{os.path.dirname(args.input_path)}/../'))
        summary_writer.add_hparams(hparams, metrics, run_name=args.hparam_save_name)
        summary_writer.flush()
    else:

        summary_writer.add_scalar('Chamfer/d2s', mean_d2s, global_step=0)
        summary_writer.add_scalar('Chamfer/s2d', mean_s2d, global_step=0)
        summary_writer.add_scalar('Chamfer/mean', over_all, global_step=0)
        summary_writer.add_scalar('Chamfer/accuracy', accuracy, global_step=0)
        summary_writer.add_scalar('Chamfer/completeness', completeness, global_step=0)
        summary_writer.add_scalar('Chamfer/f1', f1, global_step=0)
        if psnr is not None:
            summary_writer.add_scalar('Image/psnr', psnr, global_step=0)
        summary_writer.flush()
    summary_writer.close()