# VinciCoder Repurpose — VL-CoT 数据合成管线

## 项目简介

本项目将 [VinciCoder-1.6M-SFT](https://huggingface.co/datasets/DocTron-Hub/VinciCoder-1.6M-SFT) 数据集中的两个 Refine Split（`web2html_refine` 和 `chart2code_refine`）转化为两类高质量训练数据：

### 管线一：VL-CoT Coder 训练数据（`generate_vl_cot_v2.py`）

核心逻辑是 **Oracle-guided Synthesis**：
- 从原始数据中提取"有缺陷的中间代码"（Intermediate Code）和"完美的标准代码"（Ground Truth Code）。
- 对两份代码分别渲染成截图，同时喂给 `Qwen3.5-122B-A10B`。
- 模型利用 GT Code 保证输出的正确性，但推理过程（`<vlm_thought>`）必须模拟 VLM 仅凭视觉对比得出结论的过程，输出完整的修复后代码（`<vlm_action>`）。
- 支持 **Clickbait** 负样本：将 GT Code 直接作为当前代码，模型须识别出"无需修改"并输出 `DONE`。

### 管线二：VL Critic 模型训练数据（`generate_critic_data.py`）

采用 **Actor-Critic Oracle-Guided Synthesis** 架构，旨在训练一个独立的、领域通用的 Critic 模型：

- **Actor**（Qwen3.5-122B-A10B，Oracle 引导）：接收当前代码、GT Code、双图截图，生成视觉反馈（`<vlm_thought>`）、视觉失配区域的包围框（`<vlm_bbox>`）以及指令（`<vlm_action>: CONTINUE: [具体修复指令] 或 DONE`）。
- **Verifier**（Qwen3.5-122B-A10B，Oracle 引导）：独立验证 Actor 输出的准确性——包括反馈是否正确描述了视觉差异，以及修复指令是否指向正确的目标值。只有通过 Verifier 的样本（`PASS`）才会写入训练数据。
- **Critic 模型**训练时的输入/输出格式：输入为（当前代码 + 当前截图 + 目标截图），输出为 Actor 生成的（`actor_think` + `actor_vlm_thought` + `actor_vlm_action`）。

> Verifier 使用 GT Code 作为 Oracle 参考，原因：仅靠截图无法验证修复指令的目标值是否正确（例如"改为红色"vs"应改为深绿色"）。但 Verifier 的输出不会写入训练数据，不影响 Critic 模型推理时的数据分布。

---

## 目录结构

```
vincicoder_repurpose/
├── download_vincicoder_sft.py    # VinciCoder 数据集下载脚本
├── generate_vl_cot_v2.py         # 管线一：VL-CoT Coder 训练数据合成
├── generate_critic_data.py       # 管线二：VL Critic 模型训练数据合成（Actor-Critic）
├── requirements.txt              # Python 依赖
├── README.md                     # 本文档
├── models/                       # 建议将模型权重下载至此目录
│   └── Qwen3.5-122B-A10B/        # Qwen3.5-122B-A10B 模型权重
├── vincicoder_sft/               # 下载后的 Parquet 数据（由下载脚本生成）
│   ├── web2html_refine.parquet
│   └── chart2code_refine.parquet
├── inspect/                      # 管线一 --save_inspect_images 模式下的人工审查图片
│   └── <task_id>/
│       ├── intermediate_screenshot.png
│       ├── ground_truth.png
│       ├── parquet_gt_screenshot.png
│       ├── combined_screenshot.png
│       └── raw_data.json
├── inspect_critic/               # 管线二 --save_inspect_images 模式下的人工审查图片
│   └── <task_id>/
│       ├── intermediate_screenshot.png
│       ├── ground_truth.png
│       ├── parquet_gt_screenshot.png
│       ├── combined_screenshot.png
│       ├── actor_response.txt    # Actor 原始回复
│       ├── verifier_verdict.txt  # Verifier 判定结果
│       └── raw_data.json
├── vl_cot_oracle_output.jsonl    # 管线一输出（由管线一主脚本生成）
└── critic_data_output.jsonl      # 管线二输出（由管线二主脚本生成）
```

---

## 环境配置

> 推荐 Python 3.10+，依赖管理使用 `uv`。

**第一步：安装 Python 依赖**

```bash
uv pip install -r requirements.txt
```

**第二步：安装 Playwright 的 Chromium 浏览器内核**（用于渲染 HTML）

```bash
uv run playwright install chromium
```

**第三步：下载 VinciCoder 数据集**

```bash
uv run python download_vincicoder_sft.py
```

数据默认保存到 `./vincicoder_sft/`（约 10 GB，需要 HuggingFace 访问权限）。

**第四步：下载 Qwen3.5-122B-A10B 模型权重**

建议预先下载模型权重至本地，避免 vLLM 启动时临时拉取（网络不稳定时容易失败）。

模型 HuggingFace 页面：https://huggingface.co/Qwen/Qwen3.5-122B-A10B/tree/main

```bash
# 方式一：使用 huggingface_hub（推荐，支持断点续传）
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.5-122B-A10B',
    local_dir='./models/Qwen3.5-122B-A10B',
    local_dir_use_symlinks=False,
)
"
```

```bash
# 方式二：使用 huggingface-cli（需已安装 huggingface_hub）
uv run huggingface-cli download Qwen/Qwen3.5-122B-A10B \
  --local-dir ./models/Qwen3.5-122B-A10B
```

```bash
# 如果访问 HuggingFace 受限，可通过镜像站下载
HF_ENDPOINT=https://hf-mirror.com uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.5-122B-A10B',
    local_dir='./models/Qwen3.5-122B-A10B',
    local_dir_use_symlinks=False,
)
"
```

模型约 **240 GB**，请确保磁盘空间充足。下载完成后，在启动 vLLM 时指向本地路径：

```bash
# 以本地路径启动 vLLM（而非让 vLLM 自动从 HuggingFace 下载）
vllm serve ./models/Qwen3.5-122B-A10B \
  --served-model-name Qwen/Qwen3.5-122B-A10B \
  --port 7001 \
  --tensor-parallel-size 8   # 根据你的 GPU 数量调整
```

---

## 快速开始

### 管线一：生成 VL-CoT Coder 训练数据

> 前提：需要一个已启动的、兼容 OpenAI Chat API 的 vLLM 服务（参见上方启动命令）。

**仅测试渲染（不消耗 API）：**

```bash
uv run python generate_vl_cot_v2.py --render-test-only --num_html 5 --num_chart 5
```

**小规模测试（推荐首次运行）：**

```bash
uv run python generate_vl_cot_v2.py \
  --vllm-url http://localhost:7001 \
  --model-id Qwen/Qwen3.5-122B-A10B \
  --num_html 30 \
  --num_chart 30 \
  --save_inspect_images
```

**大规模完整运行：**

```bash
uv run python generate_vl_cot_v2.py \
  --vllm-url http://localhost:7001 \
  --model-id Qwen/Qwen3.5-122B-A10B \
  --num_html 2000 \
  --num_chart 2000 \
  --clickbait_ratio 0.2
```

**完整参数说明（管线一）：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--vllm-url` | `http://localhost:7003` | vLLM 服务的 Base URL |
| `--model-id` | `Qwen/Qwen3.5-122B-A10B` | 模型 ID（需与服务端一致）|
| `--num_html` | `10` | 处理的 HTML/Web 样本数（`-1` 表示全量）|
| `--num_chart` | `10` | 处理的 Chart 样本数（`-1` 表示全量）|
| `--clickbait_ratio` | `0.2` | Clickbait 负样本比例（`0~1`）|
| `--save_inspect_images` | `False` | 开启后保存渲染图片和数据到 `./inspect/` |
| `--output` | `./vl_cot_oracle_output.jsonl` | 输出 JSONL 路径 |
| `--seed` | `42` | 随机采样种子 |
| `--render-test-only` | `False` | 仅跑渲染冒烟测试，不调用 API |

---

### 管线二：生成 VL Critic 模型训练数据

**仅测试渲染（不消耗 API）：**

```bash
uv run python generate_critic_data.py --render-test-only --num_html 3 --num_chart 3
```

**小规模测试（推荐首次运行，保存 Actor/Verifier 审查文件）：**

```bash
uv run python generate_critic_data.py \
  --actor-vllm-url http://localhost:7001 \
  --verifier-vllm-url http://localhost:7001 \
  --num_html 10 \
  --num_chart 10 \
  --save_inspect_images
```

> Actor 和 Verifier 可以指向不同的 vLLM 服务端（例如跑在不同端口的同一模型），也可以指向同一服务端（默认）。

**大规模完整运行：**

```bash
uv run python generate_critic_data.py \
  --actor-vllm-url http://localhost:7001 \
  --verifier-vllm-url http://localhost:7001 \
  --num_html 2000 \
  --num_chart 2000 \
  --clickbait_ratio 0.2
```

**完整参数说明（管线二）：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--actor-vllm-url` | `http://localhost:7003` | Actor 模型 vLLM 服务 Base URL |
| `--actor-model-id` | `Qwen/Qwen3.5-122B-A10B` | Actor 模型 ID |
| `--verifier-vllm-url` | `http://localhost:7003` | Verifier 模型 vLLM 服务 Base URL |
| `--verifier-model-id` | `Qwen/Qwen3.5-122B-A10B` | Verifier 模型 ID |
| `--num_html` | `10` | 处理的 HTML/Web 样本数（`-1` 表示全量）|
| `--num_chart` | `10` | 处理的 Chart 样本数（`-1` 表示全量）|
| `--clickbait_ratio` | `0.2` | Clickbait 负样本比例（`0~1`）|
| `--save_inspect_images` | `False` | 开启后保存渲染图片、Actor 回复、Verifier 判定到 `./inspect_critic/` |
| `--output` | `./critic_data_output.jsonl` | 输出 JSONL 路径（仅含 Verifier PASS 的样本）|
| `--seed` | `42` | 随机采样种子 |
| `--render-test-only` | `False` | 仅跑渲染冒烟测试，不调用 API |

---

## 输出格式说明

### 管线一输出（`vl_cot_oracle_output.jsonl`）

每行一个 JSON 对象：

| 字段 | 说明 |
|---|---|
| `id` | 样本 ID，如 `html_0001`、`chart_cb_0002` |
| `sample_type` | `regular` 或 `clickbait` |
| `code_type` | `web` 或 `chart` |
| `model` | 调用的模型 ID |
| `intermediate_code` | 有缺陷的中间代码 |
| `gt_code` | 标准参考代码 |
| `raw_response` | 模型原始回复 |
| `think` | `<think>` 标签内容（Oracle 内部分析）|
| `vlm_thought` | `<vlm_thought>` 标签内容（模拟 VLM 的视觉推理链）|
| `vlm_action` | `<vlm_action>` 标签内容（完整修复后代码，或 `DONE`）|
| `intermediate_screenshot_b64` | 中间代码渲染截图（Base64 PNG）|
| `target_screenshot_b64` | GT 代码渲染截图（Base64 PNG）|

### 管线二输出（`critic_data_output.jsonl`）

每行一个 JSON 对象，仅包含通过 Verifier 审核（`PASS`）的样本：

| 字段 | 说明 |
|---|---|
| `id` | 样本 ID |
| `sample_type` | `regular` 或 `clickbait` |
| `code_type` | `web` 或 `chart` |
| `actor_model` | Actor 模型 ID |
| `verifier_model` | Verifier 模型 ID |
| `intermediate_code` | 有缺陷的中间代码（Critic 模型推理时的输入代码）|
| `gt_code` | 标准参考代码（仅供参考，不作为 Critic 模型输入）|
| `actor_think` | Actor 的 `<think>` 内容（Oracle 内部分析，可用于训练思考型 Critic）|
| `actor_vlm_thought` | Actor 的视觉反馈文本（描述哪里出错、该如何修改）|
| `actor_vlm_bbox` | Actor 标注的视觉失配区域包围框 `[x1, y1, x2, y2]`，或 `N/A` |
| `actor_vlm_action` | `CONTINUE: [具体修复指令]` 或 `DONE` |
| `verifier_think` | Verifier 的内部验证推理（存档用）|
| `verifier_verdict` | `PASS`（已过滤掉所有 FAIL 样本）|
| `intermediate_screenshot_b64` | 中间代码渲染截图（Base64 PNG）|
| `target_screenshot_b64` | 目标截图（Base64 PNG）|

**Critic 模型的训练输入/输出对应关系：**
- **输入**：`intermediate_code` + `intermediate_screenshot_b64` + `target_screenshot_b64`
- **输出（训练目标）**：`actor_think` + `actor_vlm_thought` + `actor_vlm_action`

---

## 核心函数说明

### 管线一（`generate_vl_cot_v2.py`）

| 函数 | 位置（约行） | 作用 |
|---|---|---|
| `render_full_page_html()` | ~253 | 用 Playwright 无头 Chromium 渲染 HTML → PNG（全页截图）|
| `render_matplotlib_chart()` | ~355 | 用 matplotlib Agg 后端执行 Python 代码 → PNG（固定尺寸）|
| `render_code()` | ~413 | 调度上面两个渲染器的入口 |
| `call_vllm_dual_image()` | ~477 | 调用 vLLM API，发送双图多模态请求，返回模型原始回复 |
| `parse_response()` | ~542 | 从模型回复中提取 `<think>`、`<vlm_thought>`、`<vlm_action>` |
| `process_regular()` | ~582 | 处理 Regular 样本：渲染 → API 调用 → 解析结果 |
| `process_clickbait()` | ~671 | 处理 Clickbait 样本：两图均为 GT 代码渲染，模型须输出 DONE |
| `save_inspect_images()` | ~424 | Inspect 模式下保存图片和 raw_data.json |
| `main()` | ~762 | CLI 入口：加载数据、采样、分批处理、写出 JSONL |

### 管线二（`generate_critic_data.py`）

| 函数 | 位置（约行） | 作用 |
|---|---|---|
| `call_vllm_actor()` | ~440 | 调用 Actor 模型（双图 + 当前代码 + GT Code）|
| `call_vllm_verifier()` | ~452 | 调用 Verifier 模型（双图 + 当前代码 + GT Code + Actor 输出）|
| `parse_actor_response()` | ~468 | 提取 `<think>`、`<vlm_thought>`、`<vlm_bbox>`、`<vlm_action>` |
| `parse_verifier_response()` | ~510 | 提取 `<think>`、`<verdict>`（缺失时返回 `FAIL: no verdict tag`）|
| `process_regular_critic()` | ~546 | Regular 样本完整 Actor-Critic 循环：渲染 → Actor → Verifier 门控 → 写出 |
| `process_clickbait_critic()` | ~640 | Clickbait 样本 Actor-Critic 循环：GT 单次渲染 → Actor → Verifier 门控 → 写出 |
| `save_inspect_images_v2()` | ~356 | 保存图片、Actor 回复文本、Verifier 判定文本、raw_data.json |
| `main()` | ~737 | CLI 入口 |

---

## API 调整指南

如果本地模型接口、API Key 或 Payload 结构与当前配置不同：

### 修改 API Key

两个脚本顶部均有：

```python
API_KEY = "sk-123456"   # ← 改成你的 API Key
```

### 修改默认服务 URL 和模型 ID

**管线一**（`generate_vl_cot_v2.py` 顶部）：

```python
DEFAULT_VLLM_URL = "http://localhost:7003"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
```

**管线二**（`generate_critic_data.py` 顶部）：

```python
DEFAULT_ACTOR_VLLM_URL    = "http://localhost:7003"
DEFAULT_VERIFIER_VLLM_URL = "http://localhost:7003"
DEFAULT_ACTOR_MODEL_ID    = "Qwen/Qwen3.5-122B-A10B"
DEFAULT_VERIFIER_MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
```

也可以直接通过 CLI 参数覆盖，无需改代码。

### 修改推理参数

两个脚本均在顶部定义了推理参数（遵循 Qwen 官方推荐配置）：

```python
TEMPERATURE        = 0.6
TOP_P              = 0.95
TOP_K              = 20
MIN_P              = 0.0
PRESENCE_PENALTY   = 0.0
REPETITION_PENALTY = 1.0
MAX_TOKENS         = 16384
```

> 如果你的接口不支持 `top_k`、`min_p`、`repetition_penalty` 等非标准字段（例如标准 OpenAI API），请在 `_call_vllm_dual_image_impl`（管线二）或 `call_vllm_dual_image`（管线一）中删去这些字段，否则会报 `422 Unprocessable Entity`。
