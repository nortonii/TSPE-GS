import open3d as o3d
import numpy as np

# 1. 加载网格
mesh = o3d.io.read_triangle_mesh("model.ply")  # 替换为你的网格文件路径
mesh.compute_vertex_normals()  # 计算法线，以便光线投射更准确

# 创建Raycasting场景
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

# 2. 设置相机参数
# 假设你已有内参和外参
K = np.array([[fx, 0, cx],  # 内参矩阵
              [0, fy, cy],
              [0, 0, 1]])

R = np.array([...])  # 旋转矩阵
t = np.array([...])  # 平移向量

# 计算相机的位置和朝向
camera_position = -np.dot(R.T, t)  # 相机位置
camera_orientation = R.T  # 相机朝向

# 3. 创建相机并生成深度图

# 创建相机的4x4外参矩阵
extrinsic = np.eye(4)
extrinsic[:3, :3] = camera_orientation
extrinsic[:3, 3] = camera_position

# 深度图的尺寸
width = 640
height = 480

# 创建光线
rays = scene.create_rays_pinhole(
    fov_deg=57.0,  # 可调视场
    eye=camera_position,
    lookat=np.array([0, 0, 0]),  # 看向场景的点
    up=np.array([0, 1, 0]),  # 上方向
    width_px=width,
    height_px=height,
)

# 进行光线投射
ans = scene.cast_rays(rays)
depth = ans['t_hit'].numpy()  # 根据光线投射结果获取深度值

# 归一化深度图（可选）
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())  # 归一化深度值

# 4. 保存深度图或可视化 
import matplotlib.pyplot as plt

plt.imshow(depth_normalized, cmap='gray')
plt.colorbar()
plt.title('Depth Map')
plt.show()

# 如果需要保存深度图到文件
depth_image = (depth_normalized * 255).astype(np.uint8)  # 转换为8位图像
plt.imsave("depth_map.png", depth_image, cmap='gray')  # 保存深度图