# 小项目一：用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

Python 程序设计课程「小项目一」的实现与实验报告。任务要求在不使用 PyTorch / OpenCLIP、不做任何参数学习的前提下，仅用 NumPy 补全 `image_encoder.py`，完整复现 OpenCLIP ViT-B/32 视觉编码器（image tower）的一次前向计算：输入预处理后的 `[B,3,224,224]` 图像，输出每张图一个 512 维单位向量，并与课程提供的独立 FP32 参考实现数值一致。

## 验收状态

`python check.py --stage all` — 8 个阶段全部通过（参考实现为独立 FP32 前向）：

| 阶段 | 输出形状 | 最大绝对误差 | 平均绝对误差 | 容差 |
| --- | --- | --- | --- | --- |
| assets | 152 个张量 | — | — | 名称/形状/哈希 |
| preprocess | (10,3,224,224) | 0 | 0 | atol 2e-6 |
| patches | (2,49,3072) | 0 | 0 | atol 2e-4 |
| tokens | (2,50,768) | 4.387e-05 | 1.335e-06 | atol 2e-4 |
| block0 | (2,50,768) | 7.147e-06 | 3.348e-07 | atol 2e-4 |
| last_tokens | (2,50,768) | 4.196e-05 | 2.633e-06 | atol 2e-4 |
| output | (10,512) | 5.662e-07 | 8.688e-08 | atol 2e-5 |
| batch | (3,512) | 4.061e-07 | 7.848e-08 | atol 2e-5 |

全部阶段 rtol 为 2e-4。输出向量的 ℓ2 范数落在 `[0.99999988, 1.0]`。

## 目录结构

```
.
├── README.md                              本文件
├── README_提交说明.md                      作业提交说明（运行步骤、验收结果、实现要点）
├── 实验报告_小项目一_李云舟2025681110.pdf     实验报告（正文 3 页）
├── main.tex                               实验报告的 LaTeX 源码（XeLaTeX 编译）
├── figs/vit_overview.png                  报告插图
└── project1-numpy-student/                课程工程
    ├── image_encoder.py                   ★ 本次实现：TODO-01 ~ TODO-06
    ├── helpers.py                         课程提供：LayerNorm / GELU / softmax / 单位化
    ├── preprocess_image.py                课程提供：缩放、裁剪、归一化
    ├── runtime.py                         课程提供：权重与图像加载
    ├── check.py                           课程提供：逐阶段数值验收
    ├── run.py                             课程提供：命令行入口
    ├── transformer_steps.py               课程提供的 2D 教学例子
    ├── config.json / PPT-CODE-MAP.txt     模型配置与讲义映射
    ├── requirements.txt                   依赖清单
    ├── assets/                            10 张测试图 + 参考输出
    └── weights/                           视觉塔权重（.npz 不入库，见下）
```

## 环境

- Python 3.9+（本项目实测 3.12.14）
- NumPy、Pillow（见 `project1-numpy-student/requirements.txt`）
- 纯 CPU 即可，不需要 GPU、PyTorch、OpenCLIP 或网络

## 运行

所有命令在 `project1-numpy-student` 目录下执行（脚本按自身位置解析权重与图像路径，与当前工作目录无关）：

```bash
cd project1-numpy-student
python -m pip install -r requirements.txt

# 逐阶段验收（推荐顺序，出错时先修最早失败的一级）
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

## 实现要点（六个 TODO）

| TODO | 方法 | 关键操作 |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | `[B,3,224,224]` → reshape `[B,3,7,32,7,32]` → transpose `(0,2,4,1,3,5)` → `[B,49,3072]` |
| TODO-02 | `MultiHeadAttention.split_heads` | `[B,N,768]` → reshape `[B,N,12,64]` → transpose `(0,2,1,3)` → `[B,12,N,64]` |
| TODO-03 | `MultiHeadAttention.__call__` | `scores = q @ k.transpose(0,1,3,2) * 1/8`；`context = coefficients @ v`；合并头后 `@ out_weight.T + out_bias` |
| TODO-04 | `TransformerBlock.__call__` | `residual = x + update`；`GELU(u @ fc1.T + b1) @ fc2.T + b2`；返回 `residual + mlp_update` |
| TODO-05 | `ImageEncoder.embed` | `patches @ patch_weight.T`（无 bias）；`concatenate([cls, projected], axis=1)`；`+ position` |
| TODO-06 | `ImageEncoder.__call__` | `tokens[:,0,:]` → `post_norm` → `@ projection`（不转置）→ `unit_vector` |

两个容易踩的坑：

1. **4 维张量不能用 `.T`**。`k.T` 会反转全部四个轴（`[B,H,N,64]` → `[64,N,H,B]`），报 `matmul ... (size 12 is different from 64)`；必须写 `k.transpose(0,1,3,2)`。
2. **阶段检查覆盖范围不同**。`--stage tokens` 只验证 `embed`，不经过注意力；注意力是否正确要跑到 `--stage block0`。

## 数据与权重

`weights/image_encoder.npz`（152 个视觉张量，约 326 MB）与 `assets/reference_outputs.npz`（参考输出）由课程附件提供，体积较大，未纳入版本控制；按课程下发的 `project1-学生版.zip` 获取后放回对应目录即可。`.gitignore` 已排除这些文件。

## 实验报告

`实验报告_小项目一_李云舟2025681110.pdf`：任务与方法、可复现性（环境、参数、运行步骤）、结果与证据（逐阶段误差表、结构与输出校验）、分析与反思（形状变化、修复的 4 维转置 bug、精度与性能、局限与改进）、本人贡献与协助说明。正文 3 页，在课程要求的 5 页限制内。
