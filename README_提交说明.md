# 提交说明：小项目一 —— 用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

本目录提交了小项目一的全部内容：代码、实验报告、仓库说明。代码部分只改动了 `project1-numpy-student/image_encoder.py`，课程提供的 `helpers.py`、`preprocess_image.py`、`runtime.py`、`check.py`、`run.py` 与测试数据均保持原样。

## 1. 文件清单

| 文件 | 说明 |
| --- | --- |
| `project1-numpy-student/image_encoder.py` | **本次实现**，补完 TODO-01 ~ TODO-06 |
| `实验报告_小项目一_李云舟2025681110.pdf` | 实验报告（正文 3 页，课程要求 ≤5 页） |
| `README.md` | 仓库说明：项目功能、代码结构、环境、运行方法、验收结果 |
| `本文件` | 提交说明：运行步骤与验收细节 |
| `main.tex` / `figs/vit_overview.png` | 课程下发的报告模板与插图（未改动） |
| `project1-numpy-student/output_tokens.npz` | `run.py` 产物（10 张测试图的输出向量），本地生成，未入库 |

## 2. 环境

- Python 3.9+（实测 3.12.14）
- NumPy、Pillow（见 `project1-numpy-student/requirements.txt`）
- 纯 CPU，不需要 GPU、PyTorch、OpenCLIP 或联网

```bash
python -m pip install -r project1-numpy-student/requirements.txt
```

## 3. 运行步骤

所有命令在 `project1-numpy-student` 目录下执行（脚本按自身位置解析权重与图像路径，与当前工作目录无关）：

```bash
cd project1-numpy-student

# 逐阶段验收（出错时先修最早失败的那一级）
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch

# 一次跑完全部阶段
python check.py --stage all

# 生成输出向量
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 4. 预期结果

`python check.py --stage all` 输出 8 行 PASS，最后一行
`Requested numerical checks passed. The reference file is never an encoder input.`

本机实测（float32，参考实现为独立 FP32 前向，rtol 均为 2e-4）：

| 阶段 | 输出形状 | 最大绝对误差 | 平均绝对误差 | 容差 |
| --- | --- | --- | --- | --- |
| assets | 152 个张量 | — | — | 名称/形状/哈希比对 |
| preprocess | (10,3,224,224) | 0 | 0 | atol 2e-6 |
| patches | (2,49,3072) | 0 | 0 | atol 2e-4 |
| tokens | (2,50,768) | 4.387e-05 | 1.335e-06 | atol 2e-4 |
| block0 | (2,50,768) | 7.147e-06 | 3.348e-07 | atol 2e-4 |
| last_tokens | (2,50,768) | 4.196e-05 | 2.633e-06 | atol 2e-4 |
| output | (10,512) | 5.662e-07 | 8.688e-08 | atol 2e-5 |
| batch | (3,512) | 4.061e-07 | 7.848e-08 | atol 2e-5 |

`check.py --stage all` 在纯 CPU 上约 11–16 s 完成，主要开销来自课程提供的逐元素 `GELU`。

## 5. 实现要点

| TODO | 方法 | 关键操作 |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | `[B,3,224,224]` → reshape `[B,3,7,32,7,32]` → transpose `(0,2,4,1,3,5)` → `[B,49,3072]` |
| TODO-02 | `MultiHeadAttention.split_heads` | `[B,N,768]` → reshape `[B,N,12,64]` → transpose `(0,2,1,3)` → `[B,12,N,64]` |
| TODO-03 | `MultiHeadAttention.__call__` | `scores = q @ k.transpose(0,1,3,2) * 1/8`；`context = coefficients @ v`；合并头后 `@ out_weight.T + out_bias` |
| TODO-04 | `TransformerBlock.__call__` | `residual = x + update`；`GELU(u @ fc1.T + b1) @ fc2.T + b2`；返回 `residual + mlp_update` |
| TODO-05 | `ImageEncoder.embed` | `patches @ patch_weight.T`（无 bias）；`concatenate([cls, projected], axis=1)`；`+ position` |
| TODO-06 | `ImageEncoder.__call__` | `tokens[:,0,:]` → `post_norm` → `@ projection`（不转置）→ `unit_vector` |

两个容易踩的坑（报告第四节有详细分析）：

1. **4 维张量不能用 `.T`**：`k.T` 会反转全部四个轴（`[B,H,N,64]` → `[64,N,H,B]`），报 `matmul ... (size 12 is different from 64)`；必须写 `k.transpose(0,1,3,2)`。
2. **阶段检查覆盖范围不同**：`--stage tokens` 只验证 `embed`，不经过注意力；注意力是否正确要跑到 `--stage block0`。

## 6. 数据与权重

- `project1-numpy-student/weights/image_encoder.npz`：152 个视觉张量，约 326 MB，由课程附件 `project1-学生版.zip` 提供，**未入库**（GitHub 不适合存放大文件）。
- `project1-numpy-student/assets/reference_outputs.npz`：参考输出，同样由课程附件提供，未入库。
- 复现时把这两个文件放回对应目录即可，其余数据（10 张测试图）已在仓库中。

## 7. 版本记录

仓库保留三个有意义的提交：

1. `chore: 导入课程下发的 Project 1 工程骨架与报告模板`
2. `feat: 实现 ViT-B/32 视觉塔六个 TODO，check.py 全阶段通过`
3. `docs: 补充提交说明、实验报告（PDF）与仓库 README`

推送到 GitHub 后请在仓库 Settings → Collaborators 中把助教与老师加入，保证可访问。
