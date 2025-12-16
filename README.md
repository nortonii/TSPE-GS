## TSPE-GS: RaDe-GS + TSPE

基于高斯溅射（Gaussian Splatting）的 3D 重建 / 渲染项目，实现并扩展了 RaDe-GS 与 TSPE 相关方法，用于在多视图场景下进行高质量重建与评估。

- **Repository**: `https://github.com/nortonii/TSPE-GS`
- **Frameworks**: PyTorch, CUDA
- **Typical tasks**: 训练、评估、网格提取（mesh extraction）、在 DTU / BlendedMVS 等数据集上的实验复现

---

## 1. 环境配置（Environment）

建议使用 Conda 创建独立环境（Python 3.8+）：

```bash
conda create -n tspe-gs python=3.8 -y
conda activate tspe-gs
```

安装基础依赖（`requirements.txt` 中列出的）：

```bash
pip install -r requirements.txt
```

---

## 2. 硬件要求（Hardware）

- **GPU**: 至少 1 块支持 CUDA 的 GPU（建议 12GB 显存或更高）
- **CUDA**: 已正确安装 CUDA 为11.8

训练脚本中会使用 `nvidia-smi` 查询显存占用，并设置 `CUDA_VISIBLE_DEVICES`；若需要使用其他 GPU，可直接修改 `train.py` 中对应代码或在外部设置环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0
```

---

## 3. 数据准备（Datasets）

本仓库包含对多种数据集的支持（从脚本名可见，如 `runbmvs*.sh`, `rundtu*.sh` 等）。典型数据集包括：

- **DTU**: 经典多视图 3D 重建数据集  
- **BlendedMVS / MVS** 系列：多视图立体重建数据集  
- 其它自定义场景（参见 `scene/` 与 `eval/` 目录）


## 4. 训练与评估（Training & Evaluation）

根目录下已经提供了若干 `.sh` 脚本来一键运行完整流程（训练 + 评估 + 网格导出），例如：

- `runtrain.sh`：默认训练脚本（可根据需要修改内部参数）
- `rundtu.sh`, `rundtueval.sh`, `rundtumesh.sh`：在 DTU 数据集上的训练 / 评估 / 网格提取
- `runbmvs.sh`, `runbmvseval.sh`, `runbmvsmesh.sh`：在 BlendedMVS / MVS 数据集上的训练 / 评估 / 网格提取
- 其他以 `run*.sh` 开头的脚本：针对不同场景或实验设置的一键运行脚本

### 4.1 基本用法示例

以 DTU 为例（假设你已经在脚本内部配置好数据路径）：

```bash
# 训练
bash rundtu.sh

# 评估
bash rundtueval.sh

# 提取重建网格
bash rundtumesh.sh
```

以 MVS 实验为例：

```bash
# 训练
bash runbmvs.sh

# 评估
bash runbmvseval.sh

# 提取重建网格
bash runbmvsmesh.sh
```

上述脚本会**自动运行完整实验流程**，包括训练和评估，并在结束时**输出最终的精度指标**（如 PSNR / 准确率等，具体见 `metric.py` 与各 `*_eval` 目录下的脚本）。

你也可以直接调用 Python 脚本，例如（根据你的实际参数进行替换）：

```bash
python train.py --config <your_config>
python render.py --scene <scene_path>
```

---

## 5. 结果示例（Example Outputs）

---

## 6. 引用（Citation）


```bibtex
@article{tspe_gs,
  title   = {TSPE-GS},
  author  = {Author et al.},
  journal = {Journal/Conference},
  year    = {2025}
}
```

---

## 7. 致谢（Acknowledgments）

本项目基于 / 参考了以下工作和代码实现（不完全列表）：

- **3D Gaussian Splatting**（原始高斯溅射代码实现）
- **RaDe-GS** 相关代码与论文
