# VinciCoder Repurpose — VL-CoT 数据合成管线

## 项目简介

本项目将 [VinciCoder-1.6M-SFT](https://huggingface.co/datasets/DocTron-Hub/VinciCoder-1.6M-SFT) 数据集中的两个 Refine Split（`web2html_refine` 和 `chart2code_refine`）转化为高质量的 **Vision-Language Chain-of-Thought (VL-CoT)** 训练数据。

核心逻辑是 **Oracle-guided Synthesis**：
- 从原始数据中提取"有缺陷的中间代码"（Intermediate Code）和"完美的标准代码"（Ground Truth Code）。
- 对两份代码分别渲染成截图，同时喂给 `Qwen3.5-122B-A10B`。
- 模型利用 GT Code 保证输出的正确性，但推理过程（`<vlm_thought>`）必须模拟 VLM 仅凭视觉对比得出结论的过程。

此外还支持 **Clickbait** 负样本：将 GT Code 直接作为当前代码，模型须识别出"无需修改"并输出 `DONE`。

---

## 目录结构

```
vincicoder_repurpose/
├── download_vincicoder_sft.py   # 数据下载脚本
├── generate_vl_cot_v2.py        # 核心合成管线（主脚本）
├── requirements.txt             # Python 依赖
├── README.md                    # 本文档
├── vincicoder_sft/              # 下载后的 Parquet 数据（由下载脚本生成）
│   ├── web2html_refine.parquet
│   └── chart2code_refine.parquet
├── inspect/                     # --save_inspect_images 模式下的人工审查图片
│   └── <task_id>/
│       ├── intermediate_screenshot.png  # 渲染 Intermediate Code 所得截图
│       ├── ground_truth.png             # 渲染 GT Code 所得截图
│       ├── parquet_gt_screenshot.png    # 数据集原始图片（用于交叉验证）
│       ├── combined_screenshot.png      # 两图左右拼合，便于对比
│       └── raw_data.json               # 原始文本数据 + 提取结果，供代码级检查
└── vl_cot_oracle_output.jsonl   # 最终输出（由主脚本生成）
```

---

## 环境配置

> 推荐 Python 3.10+。

**第一步：安装 Python 依赖**

```bash
pip install -r requirements.txt
```

**第二步：安装 Playwright 的 Chromium 浏览器内核**（用于渲染 HTML）

```bash
playwright install chromium
```

**第三步：下载数据集**

```bash
python download_vincicoder_sft.py
```

数据默认保存到 `./vincicoder_sft/`（约 10 GB，需要 HuggingFace 访问权限）。

---

## 快速开始（运行指令）

> 前提：需要一个已启动的、兼容 OpenAI Chat API 的 vLLM 服务端（如 `Qwen/Qwen3.5-122B-A10B`）。

**小规模测试（推荐第一次运行时使用）：**

```bash
python generate_vl_cot_v2.py \
  --vllm-url http://localhost:7001 \
  --model-id Qwen/Qwen3.5-122B-A10B \
  --num_html 30 \
  --num_chart 30 \
  --save_inspect_images
```

**完整参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--vllm-url` | `http://localhost:7003` | vLLM 服务的 Base URL |
| `--model-id` | `Qwen/Qwen3.5-122B-A10B` | 模型 ID（需与服务端一致）|
| `--num_html` | `10` | 处理的 HTML/Web 样本数量（`-1` 表示全量）|
| `--num_chart` | `10` | 处理的 Chart 样本数量（`-1` 表示全量）|
| `--clickbait_ratio` | `0.2` | Clickbait 负样本比例（`0~1`，默认 20%）|
| `--save_inspect_images` | `False` | 开启后保存渲染图片和原始数据到 `./inspect/` |
| `--output` | `./vl_cot_oracle_output.jsonl` | 输出 JSONL 文件路径 |
| `--seed` | `42` | 随机采样种子 |
| `--render-test-only` | `False` | 仅跑渲染冒烟测试，不调用 API |

**仅测试渲染是否正常（不消耗 API）：**

```bash
python generate_vl_cot_v2.py --render-test-only --num_html 5 --num_chart 5
```

**大规模完整跑（关闭 inspect 节省磁盘）：**

```bash
python generate_vl_cot_v2.py \
  --vllm-url http://localhost:7001 \
  --model-id Qwen/Qwen3.5-122B-A10B \
  --num_html 2000 \
  --num_chart 2000 \
  --clickbait_ratio 0.2
```

---

## 核心函数说明

| 函数 | 位置（约行） | 作用 |
|---|---|---|
| `render_full_page_html()` | ~248 | 用 Playwright 无头 Chromium 渲染 HTML → PNG（全页截图）|
| `render_matplotlib_chart()` | ~316 | 用 matplotlib Agg 后端执行 Python 代码 → PNG（固定尺寸）|
| `render_code()` | ~374 | 调度上面两个渲染器的入口 |
| `call_vllm_dual_image()` | **第 476 行** | **调用 vLLM API**，发送双图多模态请求，返回模型原始回复 |
| `parse_response()` | ~519 | 从模型回复中提取 `<think>`、`<vlm_thought>`、`<vlm_action>` |
| `process_regular()` | ~549 | 处理 Regular 样本：渲染中间代码 + GT 代码，调用 API，解析结果 |
| `process_clickbait()` | ~628 | 处理 Clickbait 样本：两图均为 GT 代码渲染，模型须输出 DONE |
| `save_inspect_images()` | ~420 | Inspect 模式下保存全套图片和 raw_data.json |
| `main()` | ~760 | CLI 入口：加载数据、采样、分批处理、写出 JSONL |

---

## API 调整指南

如果你的本地模型接口、API Key 或 Payload 结构与当前配置不同，**请定位到 `generate_vl_cot_v2.py` 第 476 行的 `call_vllm_dual_image` 函数**。

需要修改的位置汇总如下：

### 1. API Key

在文件顶部（第 61 行）修改：

```python
API_KEY = "sk-123456"   # ← 改成你的 API Key
```

### 2. 默认服务 URL 和模型 ID

在文件顶部（第 59–60 行）修改：

```python
DEFAULT_VLLM_URL = "http://localhost:7003"   # ← 改成你的服务地址
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-122B-A10B" # ← 改成你的模型 ID
```

也可以通过 CLI 参数覆盖，无需改代码：

```bash
python generate_vl_cot_v2.py --vllm-url http://your-host:8000 --model-id your/model-id
```

### 3. 请求 Header / Authorization 方式

在 `call_vllm_dual_image` 函数内（第 476 行起），找到：

```python
resp = client.post(
    url, json=payload,
    headers={"Authorization": f"Bearer {API_KEY}"},  # ← 改这里
)
```

如果你的接口用其他鉴权方式（如 `x-api-key`、无鉴权等），在这里修改 `headers` 字典即可。

### 4. 请求 Payload（模型参数）

同在 `call_vllm_dual_image` 函数内（约第 496–516 行），`payload` 字典包含所有推理参数：

```python
payload = {
    "model": model_id,
    "messages": [...],
    "max_tokens":          MAX_TOKENS,       # 16384
    "temperature":         TEMPERATURE,      # 0.6
    "top_p":               TOP_P,            # 0.95
    "top_k":               TOP_K,            # 20
    "min_p":               MIN_P,            # 0.0
    "presence_penalty":    PRESENCE_PENALTY, # 0.0
    "repetition_penalty":  REPETITION_PENALTY, # 1.0
}
```

如果你的接口不支持 `top_k`、`min_p`、`repetition_penalty` 等非标准字段（例如使用标准 OpenAI API），请删去这些字段，否则会报 `422 Unprocessable Entity`。

### 5. 图片传入方式

当前代码通过 `data:image/png;base64,...` 的 URL 格式传图（OpenAI 兼容格式）。如果你的接口要求其他格式（如单独的 `image` 字段），修改 `payload["messages"][1]["content"]` 中的图片部分即可。
