# 提交说明：Project 1 —— 用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

本目录是本次作业的完整提交内容。代码部分只改动了一个文件：`project1-numpy-student/image_encoder.py`；其余文件（`helpers.py`、`preprocess_image.py`、`runtime.py`、`check.py`、`run.py`、`transformer_steps.py`、`weights/`、`assets/`）均保持课程下发的原样。

## 1. 文件清单

| 文件 | 说明 |
| --- | --- |
| `project1-numpy-student/image_encoder.py` | **本次作业的实现**，补完 TODO-01 ~ TODO-06 |
| `main.tex` / `main.pdf` | 实验报告源码与 PDF |
| `figs/vit_overview.png` | 报告插图（课程提供） |
| `README_提交说明.md` | 本文件 |
| `project1-numpy-student/output_tokens.npz` | `run.py` 的运行产物（10 张测试图的输出向量），可选提交 |

## 2. 环境要求

- Python 3.9 或更高版本
- NumPy、Pillow（`project1-numpy-student/requirements.txt` 中列出的依赖；`imageio` 仅课堂示例需要，本实现不需要）
- 不需要 GPU、PyTorch、OpenCLIP，也不需要联网

```bash
python -m pip install -r project1-numpy-student/requirements.txt
```

## 3. 如何运行

所有命令都在 `project1-numpy-student` 目录下执行（脚本按自身位置解析权重与图像路径，不依赖当前工作目录）：

```bash
cd project1-numpy-student

# 逐阶段数值验收（推荐按顺序看，出错时先修最早失败的那一级）
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch

# 一次跑完全部阶段
python check.py --stage all

# 用 10 张课程图像生成输出向量
python run.py --output output_tokens.npz

# 用任意自己的图像生成输出向量
python run.py --images ../my_photo.jpg --output my_tokens.npz
```

## 4. 预期结果

`python check.py --stage all` 应输出 8 行 PASS，最后一行是
`Requested numerical checks passed. The reference file is never an encoder input.`

本机实测（float32，参考实现为独立 FP32 前向）：

| 阶段 | 输出形状 | 最大绝对误差 | 平均绝对误差 | 容差 |
| --- | --- | --- | --- | --- |
| assets | 152 个张量 | — | — | 张量名/形状/哈希比对 |
| preprocess | (10,3,224,224) | 0 | 0 | atol 2e-6 |
| patches | (2,49,3072) | 0 | 0 | atol 2e-4 |
| tokens | (2,50,768) | 4.387e-05 | 1.335e-06 | atol 2e-4 |
| block0 | (2,50,768) | 7.147e-06 | 3.348e-07 | atol 2e-4 |
| last_tokens | (2,50,768) | 4.196e-05 | 2.633e-06 | atol 2e-4 |
| output | (10,512) | 5.662e-07 | 8.688e-08 | atol 2e-5 |
| batch | (3,512) | 4.061e-07 | 7.848e-08 | atol 2e-5 |

全部阶段 rtol 均为 2e-4。`check.py --stage all` 在纯 CPU 上约 11–16 秒完成，主要开销来自课程提供的逐元素 `GELU`。

## 5. 实现要点（六个 TODO 对应关系）

| TODO | 方法 | 关键操作 |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | `[B,3,224,224]` → reshape `[B,3,7,32,7,32]` → transpose `(0,2,4,1,3,5)` → `[B,49,3072]` |
| TODO-02 | `MultiHeadAttention.split_heads` | `[B,N,768]` → reshape `[B,N,12,64]` → transpose `(0,2,1,3)` → `[B,12,N,64]` |
| TODO-03 | `MultiHeadAttention.__call__` | `scores = q @ k.transpose(0,1,3,2) * 1/8`；`context = coefficients @ v`；合并头后 `@ out_weight.T + out_bias` |
| TODO-04 | `TransformerBlock.__call__` | `residual = x + update`；`GELU(u @ fc1.T + b1) @ fc2.T + b2`；返回 `residual + mlp_update` |
| TODO-05 | `ImageEncoder.embed` | `patches @ patch_weight.T`（无 bias）；`concatenate([cls, projected], axis=1)`；`+ position` |
| TODO-06 | `ImageEncoder.__call__` | `tokens[:,0,:]` → `post_norm` → `@ projection`（不转置）→ `unit_vector` |

## 6. 实现中的两个注意点

1. **4 维张量不能用 `.T`**：`k.T` 会把 `[B,H,N,64]` 反转成 `[64,N,H,B]`，报 `matmul ... (size 12 is different from 64)`；必须写 `k.transpose(0,1,3,2)`，只交换最后两个轴。
2. **阶段检查的覆盖范围不同**：`--stage tokens` 只验证 `embed`，不经过注意力；注意力是否正确要跑到 `--stage block0` 才能确认。

## 7. 报告编译

`main.tex` 按模板要求使用 **XeLaTeX** 编译：

```bash
xelatex main.tex
```

模板默认中文字体为 `Noto Serif CJK SC`（Overleaf 自带）。本报告已加入字体回退：若该字体不存在，会自动使用 Windows 自带的 `SimSun`，因此在本地直接编译也能得到正确排版。
