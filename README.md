# Autonomous Driving Multimodal Pipeline

**自动驾驶图像的多模态数据处理、训练与推理** · 阿里云 PAI DSW · nuScenes · DLF 2.0 / Paimon · GPT Caption · Tongyi Embedding · Qwen2-VL · QLoRA

本仓库采用**阿里云教程风格的两篇 Jupyter Notebook**：一边阅读 Markdown 说明，一边运行代码 Cell、查看图片 / 表格 / Loss / 预测结果。**业务函数已完整内置在对应 Notebook，不依赖项目内其他 Python 模块；GitHub 可以直接预览 `.ipynb`，运行则在 DSW/JupyterLab 中完成。**

## 目录

- [1. 快速开始：两个 Notebook](#1-快速开始两个-notebook)
- [2. 阿里云用户准备工作](#2-阿里云用户准备工作)
- [3. DSW 环境安装与配置](#3-dsw-环境安装与配置)
- [4. Notebook 01：图像多模态处理](#4-notebook-01图像多模态处理)
- [5. Notebook 02：模型微调与推理](#5-notebook-02模型微调与推理)
- [6. 已验证结果与局限](#6-已验证结果与局限)
- [7. 常见问题与安全](#7-常见问题与安全)

## 1. 快速开始：两个 Notebook

| Notebook | 内容 | 运行环境 |
|---|---|---|
| **[01 · Multimodal Data Processing of Autonomous Driving Images](notebooks/01_multimodal_data_processing.ipynb)** | 下载 / 选择 nuScenes CAM_FRONT → 创建原始 Paimon 表 → GPT Caption/Metadata → 增强表 → 1152-d Embedding → Top-5 图文检索 | 阿里云 DSW，Python 3.11 |
| **[02 · Multimodal Model Training & Inference Pipeline](notebooks/02_multimodal_model_training_inference.ipynb)** | 数据导出 / 19:1 划分 → Qwen2-VL-2B 基线推理 → NF4 + FP16 → QLoRA → Loss / LoRA Adapter → 同图 Caption 对比 | DSW **V100 16GB**（本次验证） |

**推荐操作**：在 DSW JupyterLab 左侧文件浏览器中打开对应 `.ipynb`，从 Step 0 开始按顺序执行。Notebook 会逐步解释需要的环境、输入、实际调用和结果，不需要运行额外 `.py` 业务脚本。首次运行请先完成后面 2–3 节。

```text
aliyun-autonomous-driving-multimodal-pipeline/
├── README.md
├── .env.example              # 环境变量模板；实际密钥写 .env，不上传
├── .gitignore
├── requirements-data.txt
├── requirements-train.txt
├── constraints-v100.txt
└── notebooks/
    ├── 01_multimodal_data_processing.ipynb
    └── 02_multimodal_model_training_inference.ipynb
```

> 这两个 Notebook 是**自包含实现**，但并不是“零依赖”：依然需要安装第三方 Python 库，并提供 nuScenes 数据、已存在的 Paimon 表或导出的 JSONL。Notebook 02 无需运行 Notebook 01 的 Python 内存变量。

## 2. 阿里云用户准备工作

以下操作是在**阿里云控制台**完成的，区别于后续 DSW Notebook 中运行的代码。

| 步骤 | 在控制台要做的事情 | 实验说明 |
|---|---|---|
| 2.1 PAI / DSW | 开通 PAI，创建 Workspace 和 DSW 实例，选择支持的地域、云盘 | 本次使用 `cn-hangzhou`，数据与模型缓存放持久云盘 |
| 2.2 GPU 规格 | 选择 GPU 实例并查看可用规格和成本 | 已实测 **Tesla V100-SXM2-16GB**；不一定所有地域均可选择 |
| 2.3 RAM 身份 | 创建最小权限 RAM 用户/角色并授权 DLF/OSS 访问 | 长期 AK：ID+Secret；STS 临时凭证需额外 Security Token；**不要在 GitHub 存密钥** |
| 2.4 DLF 2.0 | 开通 **DLF 2.0**，创建 REST Catalog、Database、Warehouse，授予建表/读写权限 | 例：`vehicle_lake_demo`，`default` 数据库；按控制台提示启用 REST/OpenAPI |
| 2.5 网络 / OSS | 配置 DSW 所在 VPC 与 DLF 受信 VPC，OSS Bucket 和访问策略 | 遇到 `Trusted VPC`、`AccessDenied`、`DatabaseNoPermission` 时先检查权限 |
| 2.6 视觉模型 | 准备支持 OpenAI **Responses API**（图片输入）的接口和模型名 | 可使用你有权限的视觉模型服务；本项目不附带付费账号 |
| 2.7 通义 Embedding | 在百炼开通多模态 Embedding，获取 DashScope API Key / Endpoint | 本实验使用 `tongyi-embedding-vision-plus`，图片/文本统一输出 **1152 维** |
| 2.8 nuScenes 许可 | 阅读并遵守 [nuScenes 数据使用条款](https://www.nuscenes.org/) | 仓库**不包含** nuScenes 图片与模型权重 |

云端入口：[PAI](https://pai.console.aliyun.com/) · [DLF 2.0](https://dlf-next.console.aliyun.com/) · [RAM](https://ram.console.aliyun.com/) · [百炼](https://bailian.console.aliyun.com/)。控制台页面、资源与计费可能随地域及账号权限变化。

**原教程与本仓库的差异**：原阿里云案例包含 EMR Serverless Spark、DLC 多卡训练和 EAS 部署。为了降低成本，我们实际走通的是 **DSW + PyPaimon REST + OpenAI 兼容视觉 API + 百炼 Embedding + DSW 单卡 QLoRA**，未验证的 DLC / EAS 路线不计入本仓库已实现功能。

## 3. DSW 环境安装与配置

### 3.1 GPU / PyTorch 检查

在 DSW **Terminal** 中执行：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
```

本次已成功使用的主要组合：

| 项目 | 已验证配置 |
|---|---|
| Python | 3.11.14 |
| GPU | Tesla V100-SXM2-16GB (`sm_70`) |
| NVIDIA Driver | 570.211.01；驱动报告 CUDA 12.8 |
| PyTorch | `2.9.1+cu126`（CUDA runtime 12.6） |
| transformers / peft / bitsandbytes | `4.52.4` / `0.15.2` / `0.48.0` |
| 模型加载 | NF4 4-bit，FP16 计算，`attn_implementation="eager"` |

V100 不支持原生 BF16，也不要为它直接安装新 CUDA 13 构建或 FlashAttention-2。本次曾因依赖升级意外安装 `torch 2.15.0+cu132` 导致 GPU 不可用；**不要盲目使用 `--force-reinstall pypaimon`**。参考 [PyTorch 安装版本矩阵](https://pytorch.org/get-started/previous-versions/)。

如需安装对应 CUDA 12.6 PyTorch wheel（**仅当你当前环境未正确安装时**）：

```bash
python -m pip install --no-cache-dir \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

### 3.2 克隆 / 配置 `.env` / 安装依赖

```bash
git clone https://github.com/sxxa-qlp/aliyun-autonomous-driving-multimodal-pipeline.git
cd aliyun-autonomous-driving-multimodal-pipeline
cp -n .env.example .env
# 用 DSW 文本编辑器打开 .env，填写最小权限凭证和接口地址

# 先按上面 3.1 节确认 CUDA torch 完好，再安装其他依赖
python -m pip install -r requirements-data.txt -c constraints-v100.txt
python -m pip install -r requirements-train.txt -c constraints-v100.txt
python -m pip check
```

`.env` 使用 [.env.example](.env.example) 所示名称，例如：

```ini
ALIBABA_CLOUD_ACCESS_KEY_ID=你的RAM_AK_ID
ALIBABA_CLOUD_ACCESS_KEY_SECRET=你的RAM_AK_SECRET
ALIBABA_CLOUD_SECURITY_TOKEN=
VISION_API_BASE_URL=https://你的视觉API地址/v1
VISION_API_KEY=你的视觉API密钥
VISION_MODEL=你的视觉模型名称
EMBEDDING_API_URL=https://你的百炼多模态Embedding接口地址
EMBEDDING_API_KEY=你的百炼密钥
```

**长期 RAM AK 不需要 Security Token，可留空；STS 临时凭证需要同时提供三项，且会过期。** `EMBEDDING_API_URL` 是百炼原生多模态向量化接口地址，不是视觉模型的 OpenAI 兼容 Base URL。

**依赖安装后，在 Notebook 菜单选择 `Kernel → Restart Kernel`，然后从 Step 1 加载 `.env`。** 重启只会清空 Python 变量，不会删除持久云盘中的数据 / 缓存 / Adapter。两个 Notebook 在本文件内部通过 `load_dotenv(ROOT / '.env')` 和 `os.getenv` 初始化参数，**无需 `from config import settings`**。

### 3.3 首次运行开关、成本与安全

**Notebook 01 面向首次运行，7 个 `RUN_...` 开关默认全部开启：**

```python
RUN_DOWNLOAD = True
RUN_RAW_WRITE = True
RUN_CAPTION_LABELING = True
RUN_CREATE_ENRICHED = True
RUN_SAMPLE_EMBEDDING = True
RUN_VECTOR_BACKFILL = True
RUN_TEXT_RETRIEVAL = True
```

Notebook 01 按 Step 0–10 顺序执行，完成后可根据需要关闭下载、API 请求或写入。**注意：首次运行可能产生网络/存储/推理费用；Step 9 会备份后 Overwrite 实验 Paimon 表，不能并发写入，勿用于生产表。** 每个 Code Cell 前都包含目的、前置条件、结果与成本说明。默认仅对 20 张图片做视觉标注，不会自动标注完整 mini。

**Notebook 02 仍保留训练等高成本操作的独立默认开关**，本次不改动 Notebook 02。

## 4. Notebook 01：图像多模态处理

[打开完整 Notebook 01](notebooks/01_multimodal_data_processing.ipynb)

| Step | 任务 | 可以直接看到的结果 |
|---|---|---|
| 0–2 | 安装、重启 Kernel、加载 `.env`、检查 Python/GPU/凭证配置 | GPU、依赖版本和配置状态（不显示密钥） |
| 3 | nuScenes mini 下载、安全解压、选择 CAM_FRONT 连续帧 | 本地图片预览 |
| 4–5 | 连接 DLF 2.0，创建/读取原始 Paimon 表 | 表名、记录数、图片验证 |
| 6 | 视觉模型生成 Caption/Metadata，按 fileName 断点续跑 | CSV 预览、天气、道路和交通要素 |
| 7 | 构建增强 Paimon 表 `default.camfront_ai_enriched` | 标签/元数据表格 |
| 8–9 | 百炼生成 1152 维图像向量，备份并安全回填 | 向量维度、行数检查 |
| 10 | NumPy 余弦精确 Top-5，以图搜图、以文搜图 | 相似度排名和 Caption |

数据流：

```text
nuScenes / CAM_FRONT JPEG
          ↓
Paimon: default.camfront_raw_images (fileName, picture)
          ↓ Vision LLM (Caption + Metadata)
Paimon: default.camfront_ai_enriched
          ↓ tongyi-embedding-vision-plus (1152 dimensions)
Paimon enriched vector + NumPy cosine Top-5
```

注意本方案 **Paimon 无主键 Parquet Append 表**，不是 Lance 主键表；Embedding 是独立图像向量，并不是将 Caption 与图片拼接为一条向量。

## 5. Notebook 02：模型微调与推理

[打开完整 Notebook 02](notebooks/02_multimodal_model_training_inference.ipynb)

| Step | 任务 | 中间产物 |
|---|---|---|
| 0–2 | `.env` / CUDA 检查，复用 JSONL，或者从增强 Paimon 表导出并固定 19/1 划分 | `train.jsonl`, `validation.jsonl`, `split_manifest.json` |
| 3 | 预览图片与 Caption，统计训练集长度 | 图片、Caption 分布 |
| 4 | ModelScope 下载/缓存 `Qwen/Qwen2-VL-2B-Instruct`，NF4 量化加载 | Processor、模型、显存 |
| 5 | 以固定 Prompt 跑 Base Qwen2-VL | `baseline_prediction.json` |
| 6–8 | 插入 LoRA、构建 `labels=-100`、可选 Forward/Backward 测试 | 可训练参数、有效 Token 数、Loss 和梯度 |
| 9 | 19 张图、1 Epoch、梯度累积 4，保存 Adapter | `qwen2vl_lora/`, Loss |
| 10 | 同图片、同 Prompt、同生成参数进行微调后推理 | `inference_comparison.json` |

**Prompt / 图片 / 解码配置必须保持一致**，才能公平比较 Base 与 LoRA；`prepare_model_for_kbit_training` 后不可重复对同一模型调用 `get_peft_model`。

## 6. 已验证结果与局限

| 项目 | 本次 DSW 实验 |
|---|---|
| 图片 | 20 张 nuScenes CAM_FRONT 相似连续帧 |
| 划分 | 19 张训练 / 1 张验证 |
| Embedding | tongyi-embedding-vision-plus；1152 维 |
| 基础模型 | Qwen2-VL-2B-Instruct |
| 训练 | V100 16GB；4-bit NF4 + FP16；QLoRA `r=8`, `alpha=16` |
| 可训练参数 | 2,179,072（约 0.0985%） |
| 训练配置 | 1 Epoch，梯度累积 4，`1e-4` 学习率 |

**GPT 参考 Caption**：
> A city street approaches a signalized intersection with marked crosswalks, cars ahead, and pedestrians on both sidewalks. Multi-story brick buildings line the road under a cloudy daytime sky.

**微调前**：输出了建筑、车辆、行人、交通灯、天气等 7 条列表，最后因 `max_new_tokens=128` 被截断。

**微调后**：
> A busy urban street with brick buildings on both sides, traffic lights, pedestrians, and vehicles, with a cloudy sky overhead.

这证明实际的**工程数据链路 / QLoRA 风格适配**已在 DSW 走通，但相似连续帧的 19/1 验证**无法证明**跨夜间、雨天或不同道路的泛化提升。

## 7. 常见问题与安全

- **找不到变量、模型或 Processor**：Kernel 重启会清空内存；从 Notebook Step 1 顺序运行，或复用已经保存的 JSONL、模型缓存和 Adapter。
- **`NOT configured`**：检查 `.env` 文件是否放在仓库根目录，以及变量名是否与 [.env.example](.env.example) 一致。Notebook 仅确认配置非空；DLF 真正的网络/权限验证在 Notebook 01 Step 4。
- **DLF AccessDenied / Trusted VPC / DatabaseNoPermission**：检查 REST API 开通、Warehouse、OSS 与 RAM 授权、DSW 网络环境。
- **模型加载出现 CUDA 不兼容**：检查 `torch.version.cuda`、`torch.cuda.get_arch_list()` 是否包含 `sm_70`；勿在 V100 上选择 BF16 或 FlashAttention-2。
- **LoRA 重复挂载**：不要在已挂 Adapter 的模型上重复调用 `get_peft_model()`；重启 Kernel 后重新加载基础模型和 Adapter。
- **原始文件 / 密钥**：`.env`、API Key、图像、训练数据、模型、Adapter、向量缓存均不上传仓库。若密钥曾出现在公开 Git 历史，需要在云平台轮换。
- **复现验证说明**：Notebook 已通过本地格式、语法、纯函数逻辑与独立依赖检查；新重构的 Notebook 版本尚未在真实 DSW 上重跑全程。涉及 DLF、付费 API 和 V100 GPU 的路径仍需在你的 DSW 环境中确认。

官方参考：[阿里云 PAI](https://help.aliyun.com/zh/pai/) · [阿里云 DLF](https://help.aliyun.com/zh/dlf/) · [nuScenes](https://www.nuscenes.org/) · [ModelScope Qwen2-VL](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct) · [Hugging Face bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

> 许可注意：仓库没有附带 nuScenes 图片或预训练模型；相关数据与模型需遵循其原许可。仓库尚未额外指定源码许可证。