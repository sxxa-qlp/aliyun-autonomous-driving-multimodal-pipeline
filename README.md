# 阿里云 PAI 平台自动驾驶数据处理、训练与推理实践

> 文章来源：[Notion 实践笔记](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507)，整理为 GitHub 可阅读的 Markdown。原文保留了部分 DSW 实验过程及代码；仓库中的 [Notebook 01](notebooks/01_multimodal_data_processing.ipynb) 与 [Notebook 02](notebooks/02_multimodal_model_training_inference.ipynb) 是可运行版本。
>
> **公开安全提示**：RAM AccessKey、API Key 等凭证仅保存在本地 `.env`，请勿提交到 GitHub。Notion 中的控制台截图不直接复制至公共 README，配图位置提供原文入口。



## 背景介绍

选择阿里云的PAI平台学习，是因为我是阿里云ECS的重度用户，感觉aliyun平台的易用性和生态丰富性以及模型方面相比腾讯、华为、京东云都好太多了（毕竟aliyun对标的AWS）。

两个官方教程包括：

<ul>
<li>
学习并熟悉自动驾驶图象的多模态数据处理pipeline 

<b>Multimodal Data Processing of Autonomous Driving Images</b>：前视图像 → Paimon → Caption / Metadata → Embedding → 向量检索。
</li>
<li>
学习并熟悉多模态数据下的模型训练与推理

<b>Multimodal Model Training & Inference Pipeline</b>：Paimon → 固定训练/验证划分 → Qwen2-VL 基线推理 → QLoRA → 同图推理对比。
</li>
</ul>

<image source="notion-file-block://417790d8-8f7f-80f1-a189-d4977f730b9b/7db85e99-2059-4434-a041-f7e0d6fca9b7?space_id=c07790d8-8f7f-81f4-84d7-00030b99033a&name=image.png" />

注意：

1. <b>第二个教程要在第一个完成的基础上进行</b>，否则会因为没有paimon数据表导致无法训练。
2. 数据处理教程的目的是创建一个paimon来描述多模态数据，包括：
   1. Caption（场景描述）
   2. Metadata（场景结构化描述json data）
   3. embeding（图片经过embedding模型返回来的高维特征向量）

   <p>
   教程中获取这个多模态数据需要用到datawork+livy gateway+EMR访问dlf/paimon，这些是阿里云数据中台的架构【TODO：留个坑，有时间去研究下这个架构】，通过datawork平台利用livy gateway接口提交计算任务给EMR计算引擎，EMR会访问指定oss或者dlf/paimon的数据。好处是EMR内嵌了ai function可以方便的通过接口获得Caption + Metadata + embending。
   </p>

   <p>
   但缺点也很明显，这套“屠龙技”对于这个教程实在是杀鸡用牛刀了，毕竟我们重点也不在这个地方。所以我们通过如下方法获取：
   </p>

   1. caption + metadata：调用第三方视觉模型处理图片，比如“gpt-6.1-sol”
   2. embedding：调用tongyi-embedding-vision-plus模型返回向量

> <b>已验证范围</b>：2026-11-11，在阿里云 PAI DSW 的 <b>Tesla V100 16 GiB</b> 上跑通 PyTorch CUDA、Qwen2-VL-2B-Instruct NF4 量化、单卡 QLoRA 和定性推理对比；20 张 nuScenes CAM\_FRONT，训练 19 张、验证 1 张。项目源码是对交互式 Notebook 的模块化重构；<b>本仓库重构版尚未在真实 DLF / DSW 全流程重新验收</b>。无 DLC 或 EAS 依赖，原教程中的 8 卡 DLC 与 EAS 部署属于后续扩展。

<b>关键词：</b>nuScenes · Alibaba Cloud PAI DSW · DLF 2.0 / Paimon · Vision LLM · Multimodal Embedding · Qwen2-VL · QLoRA

## 目录

- [背景介绍](#背景介绍)
- [0. 准备工作](#0准备工作)
- [1. 工作流与源码导航](#1-工作流与源码导航)
- [2. 阿里云用户准备工作（控制台操作）](#2-阿里云用户准备工作控制台操作)
- [3. DSW 环境与统一配置](#3-dsw-环境与统一配置)
- [4. Part 1：多模态数据处理（Step 0–10）](#4-part-1multimodal-data-processing-of-autonomous-driving-images)
- [5. Part 2：多模态模型训练与推理](#5-part-2multimodal-model-training--inference-pipeline)
- [6. 实测结果与评价边界](#6-实测结果与评价边界)
- [7. 常见错误与排查](#7-常见错误与排查)
- [8. 安全、成本、测试与参考资料](#8-安全成本测试与参考资料)



## 0.准备工作

### 1.创建DWS

1. 在 [阿里云控制台](https://home.console.aliyun.com/) 登录。
2. 打开 [PAI 控制台](https://pai.console.aliyun.com/)，选择 <b>华东 1（杭州）</b><b><code>cn-hangzhou</code></b>，创建或选择工作空间。
3. 在工作空间中进入 <b>交互式建模（DSW）</b>，创建实例。
4. 选择兼容 CUDA 的镜像 / Python 3.11 环境；配置系统云盘。

目前的活动是aliyun PAI平台连续三个月，每月送250计算时，合计750cu。这个计算时是用来租dsw机器的计费单位，比如750cu大致可使用<b>ecs.gn7i-c8g1.2xlarge</b>，即<b>8c30g A10-24g</b>独显的dsw机器约107h，优先选这个配置的机器，因为独显高且消耗额度少。我选的时候这个机器缺货，所以就选了<b>ecs.gn6v-c8g1.2xlarge</b>的。


> 📷 **原文配图 2**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


选择不同gpu型号，可选的镜像已经是筛选之后的，也就是gpu的驱动和cuda肯定是兼容的。我们只用选择项目要求的pytorch、cuda、py版本对应的镜像就好了，很省心。比如：pytorch:2.9.1-gpu-py311-cu126-ubuntu24.04-accl。【TODO：留个坑，后面写一个pytorch cuda gpu driver的兼容链文档】


> 📷 **原文配图 3**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


<b>试用领取、额度查询</b>

[https://help.aliyun.com/zh/pai/product-overview/free-trial-guide?spm=5176.28197619.console-base\_search-panel.dtab-help\_2858359.32e73ae4VF7uOc&scm=20140722.S\_help%40%40文档%40%402858359.S\_BB2%40bl%2BRQW%40ag0%2BBB1%40ag0%2Bos0.ID\_2858359-RL\_DSW免费试用-LOC\_console\~UND\~help-OR\_ser-PAR1\_0a06dfe517945600336778720e47ba-V\_4-P0\_0-P1\_0](https://help.aliyun.com/zh/pai/product-overview/free-trial-guide?spm=5176.28197619.console-base_search-panel.dtab-help_2858359.32e73ae4VF7uOc&scm=20140722.S_help%40%40%E6%96%87%E6%A1%A3%40%402858359.S_BB2%40bl%2BRQW%40ag0%2BBB1%40ag0%2Bos0.ID_2858359-RL_DSW%E5%85%8D%E8%B4%B9%E8%AF%95%E7%94%A8-LOC_console~UND~help-OR_ser-PAR1_0a06dfe517945600336778720e47ba-V_4-P0_0-P1_0)

<b>估算大模型所需显存 </b>[https://help.aliyun.com/zh/pai/product-overview/estimation-of-the-required-video-memory-for-the-model?spm=a2c4g.11186623.help-menu-30347.d\_0\_1\_2.648d4c4bCXymYa](https://help.aliyun.com/zh/pai/product-overview/estimation-of-the-required-video-memory-for-the-model?spm=a2c4g.11186623.help-menu-30347.d_0_1_2.648d4c4bCXymYa)

<b>领取emr试用链接</b>

[https://free.aliyun.com/?utm\_source=chatgpt.com&searchKey=EMR](https://free.aliyun.com/?utm_source=chatgpt.com&searchKey=EMR)

### 2.ram给dlf授权

1. 在 [RAM 控制台](https://ram.console.aliyun.com/) 给 <b>RAM 角色授权DLF（data lake formation），id和key只在创建的时候显示一次，需要留存，在dsw里访问dlf的时候会用到</b>。
   
> 📷 **原文配图 4**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


   
> 📷 **原文配图 5**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。

2. 回到DLF首页同步ram授权，注意地域要确认和dsw区域一致。我的dsw在杭州1，这里用上海1演示。
   
> 📷 **原文配图 6**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


   <p>
   创建DLF catalogs
   </p>

   
> 📷 **原文配图 7**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


   <p>
   对这个dlf数据目录里面的paimon表授权读写操作
   </p>

   
> 📷 **原文配图 8**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


   <p>
   读写操作要分别授权，因为会从这里读原始图，会写入多模态的数据
   </p>

   
> 📷 **原文配图 9**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


### 3.准备模型密钥

1. <b>视觉标注</b>：使用支持 OpenAI Responses 图像输入的视觉模型 API<br />，比如gpt-6.1-sol
2. <b>多模态 Embedding</b>：通义 <code>tongyi-embedding-vision-plus</code>，使用百炼 <b>DashScope 原生多模态 Embedding 接口</b>，配置 <code>EMBEDDING_API_URL</code> 和 <code>EMBEDDING_API_KEY</code>；本次结果为 1152 维。注意，这里必须使用百炼提供的 <b>DashScope 接口，</b>[官方](https://docs.bailian.console.aliyun.com/zh/model-studio/embedding#:~:text=%E5%A4%9A%E6%A8%A1%E6%80%81%E7%8B%AC%E7%AB%8B%E5%90%91%E9%87%8F%E5%8C%96%E5%8A%9F%E8%83%BD%E9%9C%80%E8%A6%81%E9%80%9A%E8%BF%87%20DashScope%20SDK%20%E6%88%96%20API%20%E6%9D%A5%E8%B0%83%E7%94%A8%EF%BC%8C%E4%B8%8D%E6%94%AF%E6%8C%81%20OpenAI%20%E5%85%BC%E5%AE%B9%E6%8E%A5%E5%8F%A3%E8%B0%83%E7%94%A8%E6%88%96%E5%9C%A8%E6%8E%A7%E5%88%B6%E5%8F%B0%E7%9B%B4%E6%8E%A5%E4%BD%BF%E7%94%A8%E3%80%82)文档说了这里不支持openai兼容url<b>。</b>
   
> 📷 **原文配图 10**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。

3. <b>训练基础模型</b>：<code>Qwen/Qwen2-VL-2B-Instruct</code> 从 [ModelScope 魔搭](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct) 下载到 DSW 云盘，<b>跳转对应的代码</b>

## 1. 工作流与源码导航

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

## 2. 阿里云用户准备工作（控制台操作）

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

## 3. DSW 环境与统一配置

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

## 4. Part 1：Multimodal Data Processing of Autonomous Driving Images

### Step 0 · 获取源码、安装依赖、参数配置

```bash
git clone https://github.com/sxxa-qlp/aliyun-autonomous-driving-multimodal-pipeline.git
cd aliyun-autonomous-driving-multimodal-pipeline
cp -n .env.example .env

# 按 README 核对 torch==2.9.1+cu126、GPU sm_70；切勿盲目重装 CUDA/PyTorch
python -m pip install -r requirements-data.txt -c constraints-v100.txt
python -m pip install -r requirements-train.txt -c constraints-v100.txt
python -m pip check
```

参数配置

<i><b>这些变量在.env里面填写</b></i>


> 📷 **原文配图 11**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


<table fit-page-width="true" header-row="true">
<tr>
<td>
变量
</td>
<td>
值
</td>
</tr>
<tr>
<td>
<code>ALIBABA_CLOUD_ACCESS_KEY_ID</code>
</td>
<td>
【必填】填写你自己的 RAM AccessKey ID。
</td>
</tr>
<tr>
<td>
<code>ALIBABA_CLOUD_ACCESS_KEY_SECRET</code>
</td>
<td>
【必填】填写你自己的 RAM AccessKey Secret。
</td>
</tr>
<tr>
<td>
<code>VISION_API_BASE_URL</code>
</td>
<td>
【必填】填写视觉标注服务的 API Base URL。
</td>
</tr>
<tr>
<td>
<code>VISION_API_KEY</code>
</td>
<td>
【必填】填写视觉标注服务的 API 密钥。
</td>
</tr>
<tr>
<td>
<code>EMBEDDING_API_KEY</code>
</td>
<td>
【必填】填写对应 Embedding 服务的 API 密钥。
</td>
</tr>
<tr>
<td>
<code>VISION_MODEL</code>
</td>
<td>
【指定使用值】<code>gpt-6.1-sol</code>，用于 Caption / Metadata。模板原值为 <code>your-vision-model</code>，需要在 <code>.env</code> 中修改。
</td>
</tr>
<tr>
<td>
<code>EMBEDDING_API_URL</code>
</td>
<td>
【指定使用值】<code>https://YOUR_BAILIAN_ENDPOINT.cn-beijing.maas.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding</code>
</td>
</tr>
<tr>
<td>
<code>USE_FULL_MINI</code>
</td>
<td>
【默认】<code>false</code><br /><code>false</code>：20 张图片跑流程。<br /><code>true</code>：全部 mini CAM\_FRONT 图片。
</td>
</tr>
<tr>
<td>
<code>ALIYUN_REGION</code>
</td>
<td>
【默认】<code>cn-hangzhou</code>，与实际 DLF 服务地域一致。
</td>
</tr>
<tr>
<td>
<code>DLF_REST_URI</code>
</td>
<td>
【默认】<code>https://dlfnext.cn-hangzhou.aliyuncs.com</code>
</td>
</tr>
<tr>
<td>
<code>DLF_WAREHOUSE</code>
</td>
<td>
【默认】<code>vehicle_lake_demo</code>，如与你创建的 Warehouse 不同则修改。
</td>
</tr>
<tr>
<td>
<code>DLF_OSS_ENDPOINT</code>
</td>
<td>
【默认】<code>oss-cn-hangzhou.aliyuncs.com</code>，按实际 OSS 配置确认。
</td>
</tr>
<tr>
<td>
<code>ALIBABA_CLOUD_SECURITY_TOKEN</code>
</td>
<td>
【默认】留空；长期 RAM AccessKey 无需填写。使用 STS 临时凭证时需填写配套 Token。
</td>
</tr>
<tr>
<td>
<code>PAIMON_RAW_TABLE</code>
</td>
<td>
【默认】<code>default.camfront_raw_images</code><br />全量模式自动追加 <code>_full_mini</code> 后缀，已有该后缀时不重复追加。
</td>
</tr>
<tr>
<td>
<code>PAIMON_ENRICHED_TABLE</code>
</td>
<td>
【默认】<code>default.camfront_ai_enriched</code><br />全量模式自动追加 <code>_full_mini</code> 后缀，已有该后缀时不重复追加。
</td>
</tr>
<tr>
<td>
<code>EMBEDDING_MODEL</code>
</td>
<td>
【默认】<code>tongyi-embedding-vision-plus</code>
</td>
</tr>
<tr>
<td>
<code>EMBEDDING_VECTOR_DIM</code>
</td>
<td>
【默认】<code>1152</code>，须与模型实际输出维度一致。
</td>
</tr>
<tr>
<td>
<code>WORK_DIR</code>
</td>
<td>
【默认】<code>/mnt/systemDisk/ad_multimodal</code>，保存数据、标注 CSV、缓存和备份。
</td>
</tr>
<tr>
<td>
<code>MODEL_ID</code>
</td>
<td>
【默认】<code>Qwen/Qwen2-VL-2B-Instruct</code>，后续训练的基础模型；本 Notebook 不执行训练。
</td>
</tr>
<tr>
<td>
<code>MODEL_CACHE_DIR</code>
</td>
<td>
【默认】<code>/mnt/systemDisk/model_cache</code>，后续训练的模型缓存目录。
</td>
</tr>
<tr>
<td>
<code>SPLIT_RANDOM_SEED</code>
</td>
<td>
【默认】<code>20260520</code>，后续训练/验证数据划分的随机种子。
</td>
</tr>
<tr>
<td>
<code>VALIDATION_RATIO</code>
</td>
<td>
【默认】<code>0.05</code>，后续训练的验证集比例。
</td>
</tr>
<tr>
<td>
<code>IMAGE_MAX_TOKENS</code>
</td>
<td>
【默认】<code>256</code>，后续训练的图像 token 预算参数。
</td>
</tr>
</table>

在 JupyterLab 里通过 <b>Kernel → Restart Kernel</b> 重启

### Step 1 · 加载 <code>.env</code>（重启 Kernel 后需要执行）

<b>可调参数</b>：<code>candidates</code> 是仓库根目录的搜索位置列表，自定义目录结构时可补充路径。<code>load_dotenv(..., override=False)</code> 保留已有进程环境变量；改为 <code>True</code> 则允许 <code>.env</code> 覆盖同名变量。修改 <code>.env</code> 后建议重启 Kernel，从本步骤重新加载。

```py
# Step 1：Notebook 自包含；查找仓库根目录，仅用 .env.example 确认目录
import os
import sys
import shutil
import json
import importlib.metadata
import platform
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

cwd = Path.cwd().resolve()
candidates = [cwd, *cwd.parents,
              cwd/'aliyun-autonomous-driving-multimodal-pipeline',
              Path.home()/'workspace'/'aliyun-autonomous-driving-multimodal-pipeline']
ROOT = next((p for p in candidates if (p/'README.md').is_file()
             and (p/'.env.example').is_file() and (p/'notebooks').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('请在已 git clone 的仓库根目录或 notebooks/ 目录运行本 Notebook。')
load_dotenv(ROOT/'.env', override=False)
print('仓库:', ROOT)
print('.env 文件存在:', (ROOT/'.env').is_file())
```


> 📷 **原文配图 12**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


### Step 2 · 参数与环境检查

<b>可调参数（在 </b><b><code>.env</code></b><b> 中设置）</b>：

环境检查中的 <code>configured</code> 只表示参数非空，不代表认证成功；Step 4 才验证真实连接。

```py
# Step 2：所有参数通过环境变量 / .env 加载；不在 Notebook 内硬编码密钥
def read_boolean_environment(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value == "true"

@dataclass(frozen=True)
class Settings:
    region: str = os.getenv("ALIYUN_REGION", "cn-hangzhou")
    dlf_rest_uri: str = os.getenv("DLF_REST_URI", "https://dlfnext.cn-hangzhou.aliyuncs.com")
    dlf_warehouse: str = os.getenv("DLF_WAREHOUSE", "vehicle_lake_demo")
    oss_endpoint: str = os.getenv("DLF_OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
    access_key_id: str = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
    access_key_secret: str = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
    security_token: str = os.getenv("ALIBABA_CLOUD_SECURITY_TOKEN", "")
    raw_table: str = os.getenv("PAIMON_RAW_TABLE", "default.camfront_raw_images")
    enriched_table: str = os.getenv("PAIMON_ENRICHED_TABLE", "default.camfront_ai_enriched")
    vision_base_url: str = os.getenv("VISION_API_BASE_URL", "")
    vision_key: str = os.getenv("VISION_API_KEY", "")
    vision_model: str = os.getenv("VISION_MODEL", "your-vision-model")
    embedding_url: str = os.getenv("EMBEDDING_API_URL", "")
    embedding_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "tongyi-embedding-vision-plus")
    vector_dim: int = int(os.getenv("EMBEDDING_VECTOR_DIM", "1152"))
    work_dir: Path = Path(os.getenv("WORK_DIR", "/mnt/systemDisk/ad_multimodal"))
    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
    model_cache_dir: Path = Path(os.getenv("MODEL_CACHE_DIR", "/mnt/systemDisk/model_cache"))
    seed: int = int(os.getenv("SPLIT_RANDOM_SEED", "20260520"))
    val_ratio: float = float(os.getenv("VALIDATION_RATIO", "0.05"))
    image_max_tokens: int = int(os.getenv("IMAGE_MAX_TOKENS", "256"))
    use_full_mini: bool = read_boolean_environment("USE_FULL_MINI")

    def __post_init__(self) -> None:
        # Isolate full-run tables from smoke-test tables and vector overwrites.
        if self.use_full_mini:
            for field_name in ("raw_table", "enriched_table"):
                table_name = getattr(self, field_name)
                if not table_name.endswith("_full_mini"):
                    object.__setattr__(self, field_name, table_name + "_full_mini")

    @property
    def dataset_dir(self) -> Path:
        return self.work_dir / "vlm_training"

    @property
    def lora_output_dir(self) -> Path:
        return self.work_dir / "qwen2vl_lora"

    @property
    def nu_scenes_dir(self) -> Path:
        return self.work_dir / "nuscenes"

    @property
    def caption_csv(self) -> Path:
        filename = "caption_metadata_results_full_mini.csv" if self.use_full_mini else "caption_metadata_results.csv"
        return self.work_dir / filename

    @property
    def embedding_cache(self) -> Path:
        return self.work_dir / "embedding_cache" / self.embedding_model

    def require_dlf_credentials(self) -> None:
        if not self.access_key_id or not self.access_key_secret:
            raise RuntimeError("Set ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET (or adapt a supported role credential provider).")

    def catalog_options(self) -> dict[str, str]:
        """DLF 2.0 REST catalog options used in the verified PyPaimon 1.4 flow."""
        self.require_dlf_credentials()
        options = {
            "metastore": "rest", "uri": self.dlf_rest_uri,
            "warehouse": self.dlf_warehouse,
            "dlf.region": self.region,
            "token.provider": "dlf",
            "dlf.access-key-id": self.access_key_id,
            "dlf.access-key-secret": self.access_key_secret,
            "dlf.oss-endpoint": self.oss_endpoint,
        }
        if self.security_token:
            # Validate against the deployed PyPaimon version before using STS.
            options["dlf.security-token"] = self.security_token
        return options

settings = Settings()
def check_environment() -> bool:
    """Print diagnostics without revealing secrets or changing packages."""
    import importlib.metadata
    import platform
    print("Python:", platform.python_version())
    for name in ("numpy", "pandas", "pypaimon", "pyarrow", "torch", "transformers", "peft", "bitsandbytes", "modelscope"):
        try:
            print(f"{name:15} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name:15} [not installed]")
    print("Work dir:", settings.work_dir)
    print("Work-dir parent exists:", settings.work_dir.parent.exists())
    if settings.work_dir.parent.exists():
        print("Free GiB:", round(shutil.disk_usage(settings.work_dir.parent).free / 2**30, 1))
    for label, value in (("DLF credentials", bool(settings.access_key_id and settings.access_key_secret)),
                         ("Vision API", bool(settings.vision_base_url and settings.vision_key)),
                         ("Embedding API", bool(settings.embedding_url and settings.embedding_key))):
        print(f"{label}: {'configured' if value else 'NOT configured'}")
    try:
        import torch
        print("PyTorch CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            sm = torch.cuda.get_device_capability(0)
            print("GPU:", name, "compute capability:", sm)
            x = torch.randn(64, 64, device="cuda")
            torch.cuda.synchronize()
            print("GPU tensor test:", float((x @ x).mean()))
            return True
    except (ImportError, RuntimeError) as exc:
        print("GPU check failed:", str(exc)[:300])
    return False

check_environment()
print('数据模式:', '全量 mini CAM_FRONT (samples + sweeps)' if settings.use_full_mini else '20 张 CAM_FRONT 流程试跑')
print('Caption CSV:', settings.caption_csv)
print('Paimon Raw / Enriched:', settings.raw_table, settings.enriched_table)
print('Working dir:', settings.work_dir)
```


> 📷 **原文配图 13**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


### Step 3 · 获取 nuScenes mini，按配置选择 CAM\_FRONT 帧

下载完整 mini 后按配置选择图片；本地已备好数据时可跳过下载。

#### Step 3.1 · 下载完整 mini 数据集（下一个 Code Cell）

<b>可调参数</b>：<code>RUN_DOWNLOAD=True</code> 下载并解压，<code>False</code> 跳过整个下载/解压步骤；<code>URL</code> 是 mini 压缩包地址；<code>.env</code> 中的 <code>WORK_DIR</code> 决定压缩包和解压目录。压缩包已存在时跳过下载，但仍会解压。两种数据模式都下载完整 mini

```py
# 下载与安全解压：JupyterLab 中显示下载量/速度和解压文件数
import tarfile
import requests
from pathlib import Path
from tqdm.auto import tqdm

URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"


def download_archive(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print("压缩包已存在，跳过下载:", target)
        return target

    partial = target.with_name(target.name + ".part")
    print("开始下载:", url)
    # 当前采用覆盖下载；如 .part 已存在，会重新下载而不是断点续传。
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with partial.open("wb") as output, tqdm(
            total=total or None,
            desc="下载 nuScenes mini",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))
    partial.replace(target)
    print("下载完成:", target)
    return target


def extract_safe(archive: Path, out_dir: Path) -> None:
    """先检查全部归档路径/类型，再逐个安全解压并显示文件数量进度。"""
    root = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("开始解压:", archive)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            target = (out_dir / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive path traversal: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported tar member: {member.name}")

        for member in tqdm(members, desc="解压 nuScenes mini", unit="file"):
            tf.extract(member, path=out_dir, filter="data")

    print("解压完成:", out_dir)


RUN_DOWNLOAD = True  # False 跳过下载和解压；执行前确认磁盘空间
if RUN_DOWNLOAD:
    archive = download_archive(URL, settings.work_dir / "v1.0-mini.tgz")
    extract_safe(archive, settings.nu_scenes_dir)
    print("已解压:", settings.nu_scenes_dir)
else:
    print("下载已跳过；如已在 Paimon 保存原始图像，可直接从 Step 4 开始。")
```


> 📷 **原文配图 14**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


#### Step 3.2 · 选择全量 mini 或 20 张连续 CAM\_FRONT 图片

<b>可调参数</b>：<code>.env</code> 中 <code>USE_FULL_MINI=false</code>（默认）时，优先从 sweeps、其次从 samples 选择同日志连续 20 张图；<code>true</code> 时合并 samples 和 sweeps 的全部 CAM\_FRONT 图片，按文件名去重。全量模式使用 <code>_full_mini</code> 后缀表及独立标注 CSV。

全量模式后续生成视觉capture和Embedding 的费用更高而且需要等的时间比较长，我额度有限先走小批量跑通流程，后面等额度刷新了在跑全量的。

```py
# CAM_FRONT 搜索 + 按日志前缀与时间戳选取连续帧
import re
import io
from PIL import Image
from IPython.display import display
_TS = re.compile(r"__CAM_FRONT__(\d+)\.(?:jpg|jpeg)$", re.IGNORECASE)
def list_camfront_images(dataset_dir: Path, source: str = "sweeps") -> list[Path]:
    """Try preferred nuScenes sweeps/CAM_FRONT, fall back to samples/CAM_FRONT."""
    preferred = dataset_dir / source / "CAM_FRONT"
    paths = sorted(path for path in preferred.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"}) if preferred.is_dir() else []
    if not paths:
        matches = sorted(path for path in dataset_dir.rglob("*")
                         if path.is_file() and path.parent.name == "CAM_FRONT"
                         and path.parent.parent.name == source
                         and path.suffix.lower() in {".jpg", ".jpeg"})
        paths = matches
    if not paths:
        print(f"No {source}/CAM_FRONT images; check archive layout or try --source samples")
    return paths


def select_adjacent(paths: list[Path], limit: int = 20) -> list[Path]:
    """Stay within one filename log prefix and order by camera timestamp."""
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in paths:
        match = _TS.search(path.name)
        if match:
            prefix = path.name.split("__CAM_FRONT__", 1)[0]
            groups.setdefault(prefix, []).append((int(match.group(1)), path))
    candidates = sorted(groups.values(), key=lambda v: (-len(v), min(x[0] for x in v)))
    for group in candidates:
        if len(group) >= limit:
            return [p for _, p in sorted(group)[:limit]]
    raise ValueError(f"Need {limit} images from one CAM_FRONT log; found group lengths {[len(g) for g in candidates[:5]]}")


def read_images(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        blob = path.read_bytes()
        with Image.open(io.BytesIO(blob)) as image:
            image.verify()
        rows.append({"fileName": path.name, "picture": blob})
    return rows

selected = []
if settings.use_full_mini:
    images_by_name = {}
    for source in ('samples', 'sweeps'):
        candidates = list_camfront_images(settings.nu_scenes_dir, source=source)
        if not candidates:
            raise FileNotFoundError(f'全量模式缺少 {source}/CAM_FRONT 图片，请确认完整 mini 已解压。')
        print('相机数据源:', source, '发现:', len(candidates))
        for image_path in candidates:
            images_by_name.setdefault(image_path.name, image_path)
    selected = [images_by_name[name] for name in sorted(images_by_name)]
    print('全量 CAM_FRONT 去重后选中:', len(selected))
else:
    for source in ('sweeps', 'samples'):
        candidates = list_camfront_images(settings.nu_scenes_dir, source=source)
        try:
            selected = select_adjacent(candidates, limit=20)
            print('相机数据源:', source, '选中:', len(selected))
            break
        except ValueError as exc:
            print(source, ':', str(exc)[:150])
if selected:
    print('第一张图片:', selected[0].name)
    with Image.open(selected[0]) as im:
        display(im.convert('RGB').resize((640, 360)))
else:
    print('本地尚未找到 20 张同日志前视图片；可以复用既有 Paimon 表。')
```


> 📷 **原文配图 15**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


### Step 4 · DSW 连接 DLF 2.0 REST Catalog / Paimon

```py
# Paimon REST 连接及 Arrow 表结构；整个实现保留在 Notebook
import pyarrow as pa
import pandas as pd
RAW_SCHEMA = pa.schema([
    ("fileName", pa.string()),
    ("picture", pa.binary()),
])


ENRICHED_SCHEMA = pa.schema([
    ("id", pa.string()), ("fileName", pa.string()),
    ("picture", pa.binary()), ("caption", pa.string()),
    ("metadata", pa.string()), ("vector", pa.list_(pa.float32())),
    ("is_cloudy_candidate", pa.int32()), ("is_daytime", pa.int32()),
    ("is_nighttime", pa.int32()), ("weather", pa.string()),
    ("lighting", pa.string()), ("road_type", pa.string()),
    ("scene_type", pa.string()), ("dataset_tag", pa.int32())
])


def connect_catalog():
    from pypaimon import CatalogFactory
    return CatalogFactory.create(settings.catalog_options())


def ensure_table(catalog, name: str, arrow_schema: pa.Schema):
    from pypaimon import Schema
    database, short_name = name.split(".", 1)
    if short_name not in catalog.list_tables(database):
        schema = Schema.from_pyarrow_schema(
            arrow_schema, options={"bucket": "-1", "file.format": "parquet"}
        )
        catalog.create_table(name, schema, ignore_if_exists=False)
        print("Created:", name)
    return catalog.get_table(name)


def read_table(table):
    import pandas as pd
    rb = table.new_read_builder()
    splits = rb.new_scan().plan().splits()
    return rb.new_read().to_pandas(splits) if splits else pd.DataFrame()


def write_arrow(table, data: pa.Table, *, overwrite: bool = False) -> None:
    """Commit one Arrow batch. Overwrite is intentionally explicit."""
    builder = table.new_batch_write_builder()
    if overwrite:
        builder = builder.overwrite()
    writer, commit = builder.new_write(), builder.new_commit()
    try:
        writer.write_arrow(data)
        commit.commit(writer.prepare_commit())
    finally:
        writer.close()
        commit.close()


def insert_if_empty(table, data: pa.Table) -> bool:
    existing = read_table(table)
    if not existing.empty:
        expected_names = data.column("fileName").to_pylist()
        if existing["fileName"].duplicated().any() or set(existing["fileName"]) != set(expected_names):
            raise ValueError("Existing table filenames differ from this run; use a new table name or inspect the data manually.")
        print(f"Table not empty ({len(existing)} rows); skip append to avoid duplicates")
        return False
    write_arrow(table, data, overwrite=False)
    return True


def upload_raw_images(rows: list[dict[str, object]]) -> bool:
    catalog = connect_catalog()
    table = ensure_table(catalog, settings.raw_table, RAW_SCHEMA)
    written = insert_if_empty(table, pa.Table.from_pylist(rows, schema=RAW_SCHEMA))
    check = read_table(table)
    print(settings.raw_table, "row count:", len(check))
    if written and len(check) != len(rows):
        raise AssertionError("Raw table row count mismatch")
    return written

catalog = connect_catalog()
print('Databases:', catalog.list_databases())
print('Default tables:', catalog.list_tables('default'))
```


> 📷 **原文配图 16**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


数据湖连接正常，一个生图表，一个增强表

### Step 5 · 写入原始 Paimon 表并回读图片

原始表包含 <code>fileName</code> 和 <code>picture</code>；已有数据与当前选图一致时跳过 Append，避免重复。

<b>可调参数</b>：<code>RUN_RAW_WRITE=True</code> 写入 Step 3 选中的图片，<code>False</code> 跳过写入并读取已有表；<code>.env</code> 中 <code>PAIMON_RAW_TABLE</code> 指定表名，数据范围由 <code>USE_FULL_MINI</code> 决定。已有表文件名不匹配时会报错，不会自动覆盖

```py
RUN_RAW_WRITE = True
if RUN_RAW_WRITE:
    assert selected, '请先完成 Step 3 的 CAM_FRONT 图片选择'
    written = upload_raw_images(read_images(selected))
    print('是否新增:', written)
else:
    print('跳过原始表写入，直接读取已有 Paimon 表。')
raw_df = read_table(catalog.get_table(settings.raw_table))
if not raw_df.empty:
    if raw_df['fileName'].duplicated().any():
        raise ValueError('原始表包含重复文件名，请检查表数据。')
    if selected and set(raw_df['fileName']) != {image_path.name for image_path in selected}:
        raise ValueError('原始表与当前选图范围不同，请使用新的表名或检查数据。')
    if not settings.use_full_mini and len(raw_df) != 20:
        raise ValueError('试跑模式需要恰好 20 张图片，请检查原始表或重新运行 Step 3。')
print('原始图片行数:', len(raw_df))
if not raw_df.empty:
    display(raw_df[['fileName']].head())
    with Image.open(io.BytesIO(bytes(raw_df.iloc[0]['picture']))) as img:
        display(img.convert('RGB').resize((640,360)))
```

### Step 6 · 调用视觉 LLM 生成 Caption 和 Metadata

一次 <code>responses.create</code> 调用生成 Caption / metadata，Caption  CSV 按文件名断点续跑。

````py
# 视觉标注完整实现：图片预处理 → Responses → 解析 JSON/SSE → CSV 断点续传
import base64
import csv
import json
import re
import time
SCENE_PROMPT = """You analyze a front-facing autonomous driving camera image.
Return ONLY a JSON object containing:
caption (concise 1-2 English sentences), weather (sunny/cloudy/rainy/foggy/snowy/unknown),
lighting (daytime/nighttime/dawn/dusk/tunnel/unknown), road_type
(urban/highway/residential/parking_lot/rural/unknown), scene_type
(intersection/straight_road/curve/crosswalk_area/traffic_jam/unknown),
objects (list of strings), risks (list of strings),
is_cloudy_candidate (0/1), is_daytime (0/1), is_nighttime (0/1).
Only describe visible evidence; do not invent uncertain details. Output valid JSON only."""


FIELDS = ["fileName", "caption", "weather", "lighting", "road_type", "scene_type",
          "objects", "risks", "is_cloudy_candidate", "is_daytime", "is_nighttime"]


def image_to_data_url(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(bytes(image_bytes))) as image:
        img = image.convert("RGB")
        img.thumbnail((1280, 720))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end < start:
            raise ValueError("Vision model did not return JSON")
        result = json.loads(text[start:end+1])
    if not isinstance(result, dict):
        raise ValueError("Vision JSON must be an object")
    if not str(result.get("caption", "")).strip():
        raise ValueError("Empty caption")
    return result


def extract_response_text(response) -> str:
    """Read an OpenAI Responses object, or a custom gateway's SSE text payload.

    Some Responses-compatible proxies return `event:/data:` lines even when
    stream=False. The tutorial encountered this variation, so keep both paths.
    """
    if not isinstance(response, str):
        direct = getattr(response, "output_text", None)
        if direct:
            return str(direct).strip()
        pieces = [part.text for item in getattr(response, "output", [])
                  for part in getattr(item, "content", [])
                  if getattr(part, "type", "") == "output_text"]
        if pieces:
            return "\n".join(pieces).strip()
        raise ValueError("Responses object has no output_text")
    content = response.strip()
    if not content.startswith(("event:", "data:")):
        return content
    for line in content.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "response.failed":
            raise RuntimeError("Vision gateway reported response.failed")
        if event.get("type") == "response.completed":
            result = event.get("response", {})
            pieces = [c.get("text", "") for item in result.get("output", [])
                      for c in item.get("content", []) if c.get("type") == "output_text"]
            if pieces:
                return "\n".join(pieces).strip()
    raise ValueError("No completed response found in SSE payload")


def analyze_scene(image_bytes: bytes, *, timings: dict[str, float] | None = None) -> dict:
    """Generate one scene annotation and log image/API/JSON phase timings.

    API timing includes client creation, network transfer, gateway queueing,
    upstream inference, and waiting for the complete response (stream=False).
    """
    if not settings.vision_key or not settings.vision_base_url or settings.vision_model == "your-vision-model":
        raise RuntimeError("Configure VISION_API_BASE_URL, VISION_API_KEY, VISION_MODEL")

    t0 = time.perf_counter()
    image_url = image_to_data_url(image_bytes)
    preprocess_s = time.perf_counter() - t0
    print(f"    [1/4] 图像预处理: {preprocess_s:.2f}s", flush=True)

    print("    [2/4] 视觉模型 API: 请求中...", flush=True)
    t_api = time.perf_counter()
    from openai import OpenAI
    client = OpenAI(api_key=settings.vision_key, base_url=settings.vision_base_url, timeout=120, max_retries=1)
    response = client.responses.create(
        model=settings.vision_model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": SCENE_PROMPT},
            {"type": "input_image", "image_url": image_url, "detail": "auto"},
        ]}], stream=False, max_output_tokens=600
    )
    api_s = time.perf_counter() - t_api
    print(f"    [2/4] 视觉模型 API: {api_s:.2f}s", flush=True)

    t_parse = time.perf_counter()
    text = extract_response_text(response)
    data = parse_json_response(text)
    out = {
        "caption": str(data["caption"]).strip(),
        "weather": str(data.get("weather", "unknown")),
        "lighting": str(data.get("lighting", "unknown")),
        "road_type": str(data.get("road_type", "unknown")),
        "scene_type": str(data.get("scene_type", "unknown")),
        "objects": data.get("objects", []),
        "risks": data.get("risks", []),
        "is_cloudy_candidate": int(data.get("is_cloudy_candidate", 0)),
        "is_daytime": int(data.get("is_daytime", 0)),
        "is_nighttime": int(data.get("is_nighttime", 0)),
    }
    if not isinstance(out["objects"], list) or not isinstance(out["risks"], list):
        raise ValueError("objects / risks should be lists")
    parse_s = time.perf_counter() - t_parse
    print(f"    [3/4] Caption/Metadata 解析: {parse_s:.2f}s", flush=True)

    if timings is not None:
        timings.update(preprocess_s=preprocess_s, api_s=api_s, parse_s=parse_s)
    return out


def annotate_to_csv(records, path: Path) -> int:
    """Resume by fileName; print phase timings and persist each successful result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as f:
            existing = {row["fileName"]: row for row in csv.DictReader(f)}
    done = 0
    skipped = 0
    total = len(records) if hasattr(records, "__len__") else None
    batch_started = time.perf_counter()

    for index, record in enumerate(records, 1):
        filename = str(record["fileName"])
        progress = f"[{index}/{total}]" if total is not None else f"[{index}]"
        if filename in existing:
            skipped += 1
            print(f"{progress} {filename}: CSV 已存在，跳过", flush=True)
            continue

        print(f"{progress} 开始处理: {filename}", flush=True)
        image_started = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            analysis = analyze_scene(bytes(record["picture"]), timings=timings)

            t_csv = time.perf_counter()
            row = {"fileName": filename, **analysis}
            row["objects"] = json.dumps(row["objects"], ensure_ascii=False)
            row["risks"] = json.dumps(row["risks"], ensure_ascii=False)
            existing[filename] = row
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(existing.values())
            csv_s = time.perf_counter() - t_csv
            print(f"    [4/4] CSV 保存: {csv_s:.2f}s", flush=True)
            done += 1
            print(f"    单张总耗时: {time.perf_counter() - image_started:.2f}s"
                  f"（API: {timings['api_s']:.2f}s）", flush=True)
            print(f"    Caption: {analysis['caption']}", flush=True)
        except Exception as exc:
            print(f"    处理失败（已耗时 {time.perf_counter() - image_started:.2f}s，"
                  f"错误类型: {type(exc).__name__}），该图片未完成标注", flush=True)
            raise

    print(f"新标注: {done} 张；跳过已有: {skipped} 张；CSV 总记录: {len(existing)}；"
          f"本轮耗时: {time.perf_counter() - batch_started:.2f}s", flush=True)
    return done


RUN_CAPTION_LABELING = True
if RUN_CAPTION_LABELING:
    assert not raw_df.empty, '请先准备原始图片表'
    annotate_to_csv(raw_df.to_dict('records'), settings.caption_csv)
if settings.caption_csv.is_file():
    labels_df = pd.read_csv(settings.caption_csv)
    display(labels_df[['fileName','caption','weather','lighting','road_type']].head())
else:
    labels_df = None
    print('未找到 Caption CSV；需要标注时将 RUN_CAPTION_LABELING 改 True。')
````


> 📷 **原文配图 17**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


### Step 7 · 创建多模态增强表并写入

<code>camfront_ai_enriched</code> 包含图片、GPT Caption、Metadata、<code>vector</code> 字段和 <code>dataset_tag=1</code>。这是<b>无主键的 Parquet Append 表</b>，只在空表时写入。这里其实还差embedding vector还没写进入。

<b>可调参数</b>：<code>RUN_CREATE_ENRICHED=True</code> 将原始图片与 CSV 标注合并并写入增强表，<code>False</code> 跳过创建/写入并读取已有表；<code>.env</code> 中 <code>PAIMON_ENRICHED_TABLE</code> 指定表名。CSV 文件名集合必须与原始表完全一致；已有增强表也会校验文件名集合，不匹配时报错

```py
# Schema / 数据转换 / 建表 / 写入
import uuid
def make_enriched_records(raw_df: pd.DataFrame, labels: list[dict]) -> list[dict]:
    labels_by_name = {str(row["fileName"]): row for row in labels}
    if len(labels_by_name) != len(labels):
        raise ValueError("Duplicate caption filenames")
    if raw_df["fileName"].duplicated().any():
        raise ValueError("Duplicate source filenames")
    if set(raw_df["fileName"]) != set(labels_by_name):
        raise ValueError("Raw image and caption fileName sets differ")
    rows = []
    for row in raw_df.to_dict("records"):
        name = str(row["fileName"])
        meta = labels_by_name[name]
        weather, lighting = meta.get("weather", "unknown"), meta.get("lighting", "unknown")
        objects = meta.get("objects", [])
        risks = meta.get("risks", [])
        objects = json.loads(objects) if isinstance(objects, str) else objects
        risks = json.loads(risks) if isinstance(risks, str) else risks
        metadata = {
            "weather": weather, "lighting": lighting, "road_type": meta.get("road_type", "unknown"),
            "scene_category": meta.get("scene_type", "unknown"), "objects": objects, "risks": risks,
        }
        rows.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)), "fileName": name,
            "picture": bytes(row["picture"]), "caption": str(meta["caption"]),
            "metadata": json.dumps(metadata, ensure_ascii=False), "vector": None,
            "is_cloudy_candidate": int(str(weather).lower() == "cloudy"),
            "is_daytime": int(str(lighting).lower() == "daytime"),
            "is_nighttime": int(str(lighting).lower() == "nighttime"),
            "weather": weather, "lighting": lighting, "road_type": meta.get("road_type", "unknown"),
            "scene_type": f"{lighting}_{weather}", "dataset_tag": 1,
        })
    return rows


def create_enriched_table() -> None:
    import pyarrow as pa
    # Helpers defined earlier in Step 4
    cat = connect_catalog()
    source = read_table(cat.get_table(settings.raw_table))
    if source.empty:
        raise ValueError("Raw image table empty")
    with settings.caption_csv.open(encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    rows = make_enriched_records(source, labels)
    table = ensure_table(cat, settings.enriched_table, ENRICHED_SCHEMA)
    insert_if_empty(table, pa.Table.from_pylist(rows, schema=ENRICHED_SCHEMA))
    print("Enriched rows:", len(read_table(table)))

RUN_CREATE_ENRICHED = True
if RUN_CREATE_ENRICHED:
    assert labels_df is not None, '需先标注或提供 CSV'
    create_enriched_table()
else:
    print('跳过建表 / 写入')
enriched_database, enriched_table_name = settings.enriched_table.split('.', 1)
if enriched_table_name in catalog.list_tables(enriched_database):
    enriched_df = read_table(catalog.get_table(settings.enriched_table))
    print('增强数据行数:',len(enriched_df))
    if not enriched_df.empty:
        display(enriched_df[['fileName','caption','weather','lighting','road_type']].head())
else:
    enriched_df = pd.DataFrame()
    print('增强表尚不存在：需要先完成 Step 7')
```


> 📷 **原文配图 18**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


### Step 8 · 生成 1152 维图像/文本 Embedding

使用百炼 DashScope 原生多模态接口，<b>不是 OpenAI embeddings 接口</b>；图像和文本分别编码但可做跨模态比较。按内容哈希缓存减少重复调用。

<b>可调参数</b>：<code>RUN_SAMPLE_EMBEDDING=True</code> 对增强表第一张图片生成或读取缓存向量，<code>False</code> 跳过单图验证；

```py
# DashScope 向量服务、SHA256 本地缓存
import numpy as np
import hashlib
def _embed(content: dict) -> np.ndarray:
    if not settings.embedding_url or not settings.embedding_key:
        raise RuntimeError("Set EMBEDDING_API_URL and EMBEDDING_API_KEY")
    response = requests.post(
        settings.embedding_url,
        headers={"Authorization": f"Bearer {settings.embedding_key}", "Content-Type": "application/json"},
        json={"model": settings.embedding_model, "input": {"contents": [content]}},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code"):
        raise RuntimeError(f"Embedding API error code: {payload['code']}; {payload.get('message', '')}")
    entries = payload.get("output", {}).get("embeddings", [])
    if len(entries) != 1:
        raise ValueError(f"Expected one embedding, received {len(entries)}")
    vector = np.asarray(entries[0]["embedding"], dtype=np.float32)
    if vector.shape != (settings.vector_dim,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid embedding: {vector.shape}, expected {settings.vector_dim}")
    return vector


def _cached(key: bytes, kind: str, produce) -> np.ndarray:
    settings.embedding_cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(settings.embedding_model.encode() + kind.encode() + key).hexdigest()
    path = settings.embedding_cache / (digest + ".npy")
    if path.is_file():
        vector = np.load(path, allow_pickle=False)
    else:
        vector = produce()
        temp = path.with_suffix(".tmp.npy")
        np.save(temp, vector)
        temp.replace(path)
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (settings.vector_dim,) or not np.isfinite(vector).all():
        raise ValueError("Invalid cached vector")
    return vector


def image_embedding(picture: bytes) -> np.ndarray:
    raw = bytes(picture)
    return _cached(raw, "image", lambda: _embed({"image": image_to_data_url(raw)}))


def text_embedding(query: str) -> np.ndarray:
    if not query.strip():
        raise ValueError("Query must not be empty")
    return _cached(query.encode("utf-8"), "text", lambda: _embed({"text": query}))

RUN_SAMPLE_EMBEDDING = True
if RUN_SAMPLE_EMBEDDING:
    assert not enriched_df.empty, '请先准备增强表'
    vector = image_embedding(enriched_df.iloc[0]['picture'])
    print('向量维度:', vector.shape, '范数:', np.linalg.norm(vector))
else:
    print('跳过 Embedding API 调用')
```


> 📷 **原文配图 19**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


每张图1152维的embedding向量

### Step 9 · embedding回填 Paimon 向量字段

无主键 Append 表不支持通过追加同 ID 行来更新。先将<b>整表备份为 Parquet</b>，计算向量，再明确 Overwrite，最后重新读取校验。期间不要并发写入。

<b>可调参数</b>：<code>RUN_VECTOR_BACKFILL=True</code> 执行批量向量回填，<code>False</code> 跳过；调用参数 <code>confirm_overwrite=True</code> 明确允许覆盖，设为 <code>False</code> 会在需要覆盖时中止。备份保存在 <code>.env</code> 的 <code>WORK_DIR/backups</code>。执行顺序为备份、生成向量、Overwrite、回读校验；已有完整向量时跳过，空表/重复 ID/部分已有向量时中止。<b>会修改云端数据并可能产生大量 Embedding 费用</b>，请确认目标表、无并发写入且备份空间充足。

```py
# 只在明确开启、已确认无并发写入时执行 destructive overwrite
from datetime import datetime, timezone
def _vector_present(value) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def backfill_vectors(*, confirm_overwrite: bool = False) -> Path | None:
    """Backup, vectorize all, and overwrite only after explicit confirmation."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    # Helpers defined earlier in Step 4
    cat = connect_catalog()
    table = cat.get_table(settings.enriched_table)
    original = read_table(table)
    if original.empty or original["id"].duplicated().any():
        raise ValueError("Empty / duplicate-ID enriched table; inspect manually")
    has = [_vector_present(v) for v in original["vector"]]
    if all(has):
        for v in original["vector"]:
            if len(v) != settings.vector_dim:
                raise ValueError("Unexpected vector dimension")
        print("Already embedded; no overwrite needed")
        return None
    if any(has):
        raise ValueError("Partially embedded table; inspect manually before overwrite")
    if not confirm_overwrite:
        raise RuntimeError("Overwrite required; rerun with --confirm-overwrite after reviewing table/backup policy")
    backup_dir = settings.work_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"enriched_before_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.parquet"
    pq.write_table(pa.Table.from_pandas(original, schema=ENRICHED_SCHEMA, preserve_index=False), backup_path)
    print("Backup:", backup_path)
    updated = original.copy()
    updated["vector"] = [image_embedding(bytes(pic)).tolist() for pic in updated["picture"]]
    if any(len(v) != settings.vector_dim for v in updated["vector"]):
        raise RuntimeError("Vector dimension mismatch; aborting write")
    write_arrow(table, pa.Table.from_pandas(updated, schema=ENRICHED_SCHEMA, preserve_index=False), overwrite=True)
    result = read_table(table)
    if len(result) != len(original) or any(not _vector_present(v) or len(v) != settings.vector_dim for v in result["vector"]):
        raise RuntimeError("Post-overwrite verification failed; restore using backup")
    print("Backfill verified:", len(result), "rows x", settings.vector_dim)
    return backup_path

RUN_VECTOR_BACKFILL = True  # 谨慎：会覆盖增强表
if RUN_VECTOR_BACKFILL:
    backfill_vectors(confirm_overwrite=True)
    enriched_df = read_table(catalog.get_table(settings.enriched_table))
else:
    print('已跳过向量覆盖写入')
```


> 📷 **原文配图 20**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


吧embedding回填到paimon表里面，至此，enrichment table就完成了

### Step 10 · 以图搜图 / 以文搜图：余弦 Top-5

NumPy 进行精确余弦相似度检索，未使用 Milvus/ANN，这个就是；文本查询在缓存未命中时调用百炼 Embedding，会产生 API 调用费用。

#### Step 10.1 · 以图搜图（下一个 Code Cell）

<b>可调参数</b>：<code>q = vectors[0]</code> 中的索引选择查询图片；更换索引时同步修改 <code>exclude_index=0</code>，以排除查询图片自身。<code>cosine_topk(..., k=5)</code> 的 <code>k</code> 控制返回数量。此步骤使用已有图像向量，不调用 API、不改写 Paimon。

```py
import io
import matplotlib.pyplot as plt
from PIL import Image

def cosine_topk(vectors: np.ndarray, query_vector: np.ndarray, k: int = 5,
                exclude_index: int | None = None) -> list[tuple[int, float]]:
    values = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if values.ndim != 2 or query.shape != (values.shape[1],):
        raise ValueError("Embedding dimensions differ")
    norms = np.linalg.norm(values, axis=1)
    q_norm = np.linalg.norm(query)
    if np.any(norms == 0) or q_norm == 0:
        raise ValueError("Zero-norm vector")
    scores = (values / norms[:, None]) @ (query / q_norm)
    if exclude_index is not None:
        scores[exclude_index] = -np.inf
    order = np.argsort(-scores, kind="stable")
    return [(int(i), float(scores[i])) for i in order[:min(k, len(order))] if np.isfinite(scores[i])]

if enriched_df.empty or any(not _vector_present(v) for v in enriched_df.get('vector', [])):
    print('增强表缺少图像向量；请先完成 Step 8–9')
else:
    vectors = np.stack([np.asarray(v, dtype=np.float32) for v in enriched_df['vector']])
    q = vectors[0]
    for rank,(idx,score) in enumerate(cosine_topk(vectors,q,k=5,exclude_index=0),1):
        print(f'{rank}. cosine={score:.4f}',enriched_df.iloc[idx]['fileName'])


if (
    enriched_df.empty
    or 'vector' not in enriched_df.columns
    or any(not _vector_present(v) for v in enriched_df['vector'])
):
    print('增强表缺少图像向量；请先完成 Step 8–9')

else:
    # 1. 读取全部图片 Embedding
    vectors = np.stack([
        np.asarray(v, dtype=np.float32)
        for v in enriched_df['vector']
    ])

    # 2. 选择第一张图片作为目标图
    query_idx = 0
    q = vectors[query_idx]

    # 3. 检索最相似的 5 张图片（排除自身）
    results = cosine_topk(
        vectors, q, k=5, exclude_index=query_idx
    )

    # 4. 目标图片 + Top-5 检索结果
    items = [(query_idx, None)] + results

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    for pos, (idx, score) in enumerate(items):
        row = enriched_df.iloc[idx]

        # 从 Paimon picture 字段读取图片二进制
        with Image.open(io.BytesIO(bytes(row['picture']))) as img:
            axes[pos].imshow(img.convert('RGB'))

        axes[pos].axis('off')

        if score is None:
            title = "Query Image"
        else:
            title = f"Top {pos} | Cosine: {score:.4f}"

        axes[pos].set_title(
            title + "\n" + str(row['fileName'])[-42:],
            fontsize=10
        )

    # 隐藏没有使用的子图
    for ax in axes[len(items):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # 5. 同时打印检索结果及 Caption
    print("\n========== Top-5 检索结果 ==========")

    for rank, (idx, score) in enumerate(results, 1):
        row = enriched_df.iloc[idx]
        print(f"\nTop {rank} | cosine={score:.4f}")
        print("文件名:", row['fileName'])
        print("Caption:", row['caption'])
```


> 📷 **原文配图 21**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


<p />

#### Step 10.2 · 以文搜图：自然语言查询 Top-5

<b>可调参数</b>：<code>RUN_TEXT_RETRIEVAL=True</code> 执行文本检索，<code>False</code> 跳过；<code>query</code> 设置自然语言查询内容，<code>cosine_topk(..., k=5)</code> 的 <code>k</code> 控制返回数量。使用 Step 8 的 Embedding 配置和 Step 9 的图像向量；文本向量也调用的是tongyi的vison模型，输出维度也是1152，这样才能在同维语义空间通过余弦比较相似

```py

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

# ==================== Step 10：以文搜图 ====================

RUN_TEXT_RETRIEVAL = True

# 目标文本：希望从自动驾驶图片中检索出的场景
query = (
    "A cloudy daytime urban intersection with "
    "pedestrians, cars and traffic lights."
)

if RUN_TEXT_RETRIEVAL:

    # 1. 检查增强表和 Embedding 是否准备完成
    # Step 9 已将每张图片的 1152 维向量回填到 Paimon
    assert not enriched_df.empty, "增强表为空，请先完成 Step 7"
    assert 'vector' in enriched_df.columns, "缺少 vector 字段"
    assert all(
        _vector_present(v) for v in enriched_df['vector']
    ), "存在未生成 Embedding 的图片，请先完成 Step 9"

    # 2. 读取所有图片向量
    # 如果有 20 张图片：
    # vectors.shape = (20, 1152)
    vectors = np.stack([
        np.asarray(v, dtype=np.float32)
        for v in enriched_df['vector']
    ])

    # 3. 将查询文本转换成 1152 维向量
    # 调用 tongyi-embedding-vision-plus（已有缓存则直接读取）
    # qvec.shape = (1152,)
    qvec = text_embedding(query)

    print("Query:", query)
    print("图片总数:", len(enriched_df))
    print("图片向量维度:", vectors.shape)
    print("文本向量维度:", qvec.shape)

    # 4. 计算文本向量与所有图片向量的余弦相似度
    # cosine_topk 内部先对向量做 L2 归一化，
    # 再通过矩阵乘法计算相似度并降序排列。
    #
    # 返回 [(图片行索引, 余弦相似度), ...]
    results = cosine_topk(vectors, qvec, k=5)

    # 5. 输出 Top-5 检索结果表格
    result_df = pd.DataFrame([
        {
            'rank': rank,
            'score': score,
            'fileName': enriched_df.iloc[idx]['fileName'],
            'caption': enriched_df.iloc[idx]['caption']
        }
        for rank, (idx, score) in enumerate(results, 1)
    ])

    print("\n========== Top-5 检索结果 ==========")
    display(result_df)

    # 6. 可视化 Top-5 图片
    # 目标为文本 Query，因此将其展示在图表顶部。
    # 每张图显示排名、余弦相似度和文件名。
    if results:
        cols = min(3, len(results))
        rows = (len(results) + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(5.8 * cols, 4.5 * rows),
            squeeze=False
        )
        axes = axes.flatten()

        for rank, (idx, score) in enumerate(results, 1):
            row = enriched_df.iloc[idx]
            ax = axes[rank - 1]

            # 从 Paimon picture 字段读取 JPEG 图片
            with Image.open(
                io.BytesIO(bytes(row['picture']))
            ) as img:
                ax.imshow(img.convert('RGB'))

            ax.axis('off')

            ax.set_title(
                f"Top {rank} | Cosine: {score:.4f}\n"
                f"{str(row['fileName'])[-45:]}",
                fontsize=10
            )

        # 隐藏未使用的子图位置
        for ax in axes[len(results):]:
            ax.axis('off')

        # 图表顶部展示查询文本
        fig.suptitle(
            "Text-to-Image Retrieval\n" + query,
            fontsize=13,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()
        plt.show()

else:
    print('以文搜图关闭；需要时将 RUN_TEXT_RETRIEVAL 改 True。')

```


> 📷 **原文配图 22**：截图保留在 [Notion 原文](https://app.notion.com/p/aliyun-PAI-417790d88f7f80778eabc681a20c9507) 中（不在公开仓库复制可能包含账号或控制台信息的截图）。


文搜图的相似性挺低的

### Part 1 总结与下一步

- 已验证处理链路：nuScenes CAM\_FRONT → Paimon 原始表 → GPT Caption/Metadata → 增强 Paimon 表 → Tongyi 1152-d Embedding → Top-5 检索。
- Notebook 02 可以直接读取 Paimon 增强表；如果已有 <code>vlm_training/train.jsonl</code>，则优先复用它，保持之前的验证样本与基线预测不变。
- <b>本 Notebook 的全部函数在本文件中定义</b>，没有 <code>from multimodal_data_processing import ...</code>。

## 5. Part 2：Multimodal Model Training & Inference Pipeline

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

## 6. 实测结果与评价边界

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

## 7. 常见错误与排查

- **找不到变量、模型或 Processor**：Kernel 重启会清空内存；从 Notebook Step 1 顺序运行，或复用已经保存的 JSONL、模型缓存和 Adapter。
- **`NOT configured`**：检查 `.env` 文件是否放在仓库根目录，以及变量名是否与 [.env.example](.env.example) 一致。Notebook 仅确认配置非空；DLF 真正的网络/权限验证在 Notebook 01 Step 4。
- **DLF AccessDenied / Trusted VPC / DatabaseNoPermission**：检查 REST API 开通、Warehouse、OSS 与 RAM 授权、DSW 网络环境。
- **模型加载出现 CUDA 不兼容**：检查 `torch.version.cuda`、`torch.cuda.get_arch_list()` 是否包含 `sm_70`；勿在 V100 上选择 BF16 或 FlashAttention-2。
- **LoRA 重复挂载**：不要在已挂 Adapter 的模型上重复调用 `get_peft_model()`；重启 Kernel 后重新加载基础模型和 Adapter。

## 8. 安全、成本、测试与参考资料

- **原始文件 / 密钥**：`.env`、API Key、图像、训练数据、模型、Adapter、向量缓存均不上传仓库。若密钥曾出现在公开 Git 历史，需要在云平台轮换。
- **复现验证说明**：Notebook 已通过本地格式、语法、纯函数逻辑与独立依赖检查；新重构的 Notebook 版本尚未在真实 DSW 上重跑全程。涉及 DLF、付费 API 和 V100 GPU 的路径仍需在你的 DSW 环境中确认。

官方参考：[阿里云 PAI](https://help.aliyun.com/zh/pai/) · [阿里云 DLF](https://help.aliyun.com/zh/dlf/) · [nuScenes](https://www.nuscenes.org/) · [ModelScope Qwen2-VL](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct) · [Hugging Face bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

> 许可注意：仓库没有附带 nuScenes 图片或预训练模型；相关数据与模型需遵循其原许可。仓库尚未额外指定源码许可证。

**成本与数据安全补充**：`USE_FULL_MINI=true` 会扩大 Caption 和 Embedding API 调用量；执行前需确认额度、速率、云盘空间。Notebook 01 Step 9 采用备份后 Overwrite 的实验表向量回填策略，不要对生产表直接执行。

**内容范围**：Notion 当前正文详细记录到 Part 1 Step 10；Part 2、实测结果与常见问题沿用本仓库原 README 的已有说明。上述配图链接指向 Notion 页面，GitHub 中并未托管截图文件。
