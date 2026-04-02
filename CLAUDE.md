# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **VL-CoT Data Synthesis Pipeline** that transforms the VinciCoder-1.6M-SFT dataset (`web2html_refine` and `chart2code_refine` splits) into Vision-Language Chain-of-Thought (VL-CoT) training data using **Oracle-guided Synthesis**.

Core idea: Given a flawed intermediate code and ground-truth code, render both to screenshots, send the pair to a VLM, and have it generate a visual reasoning trace (`<vlm_thought>`) that simulates debugging by comparing screenshots — without explicitly revealing the ground-truth code in the prompt.

## Setup

```bash
# Install dependencies (use uv per global guidelines)
uv pip install -r requirements.txt

# Install Playwright's headless Chromium (required for HTML rendering)
uv run playwright install chromium

# Download the source datasets from HuggingFace
uv run python download_vincicoder_sft.py
```

## Running the Pipeline

```bash
# Quick test (30 + 30 samples, saves inspection images)
uv run python generate_vl_cot_v2.py \
  --vllm-url http://localhost:7001 \
  --model-id Qwen/Qwen3.5-122B-A10B \
  --num_html 30 --num_chart 30 \
  --save_inspect_images

# Render smoke-test only (no API calls needed)
uv run python generate_vl_cot_v2.py --render-test-only --num_html 5 --num_chart 5

# Full scale run (2000 + 2000 samples, 20% clickbait)
uv run python generate_vl_cot_v2.py \
  --vllm-url http://localhost:7001 \
  --model-id Qwen/Qwen3.5-122B-A10B \
  --num_html 2000 --num_chart 2000 \
  --clickbait_ratio 0.2
```

Key CLI arguments: `--vllm-url`, `--model-id`, `--num_html`, `--num_chart`, `--clickbait_ratio` (default 0.2), `--seed` (default 42), `--output` (default `./vl_cot_oracle_output.jsonl`), `--save_inspect_images`, `--render-test-only`.

## Architecture

### Files
- `download_vincicoder_sft.py` — Downloads parquet files from HuggingFace to `./vincicoder_sft/`
- `generate_vl_cot_v2.py` — Main 958-line pipeline script; entry point is `main()` at line 762

### Pipeline Flow

```
Parquet files → Sample & split into Regular/Clickbait batches
  → For each sample:
      Regular: render(intermediate_code) + render(gt_code) → dual-image VLM call
      Clickbait: render(gt_code) used as BOTH images → model must output "DONE"
  → Parse <think>, <vlm_thought>, <vlm_action> from response
  → Write to JSONL (with base64-encoded screenshots embedded)
```

### Key Subsystems (all in `generate_vl_cot_v2.py`)

**Rendering** (lines 248–418):
- HTML: `render_full_page_html()` — Playwright + headless Chromium, 1280×720 viewport, full-page capture, up to 5 retries
- Charts: `render_matplotlib_chart()` — executes Python with `exec()`, forces 10×8in @ 100 DPI (1000×800px), matplotlib Agg backend
- `render_code()` dispatches to the correct renderer based on `code_type`

**API Client** (lines 477–536):
- `call_vllm_dual_image()` — OpenAI-compatible Chat Completions format, sends two base64 PNG images, 3 retries with exponential backoff, 240s timeout
- HTTP proxies are explicitly cleared before each request (lines 49–53)
- Default API key: `sk-123456` (change at line 48 or override via env)

**Response Parsing** (lines 542–576):
- `parse_response()` extracts `<think>`, `<vlm_thought>`, `<vlm_action>` tags
- Handles 3 edge cases: standard format, collapsed tags (model merges `<think>` into `<vlm_thought>`), and truncated responses

**Sample Processing** (lines 582–756):
- `process_regular()` — renders two different images and calls VLM to produce a correction reasoning trace
- `process_clickbait()` — uses identical images (GT rendered twice); validates that `vlm_action == "DONE"`

**Inspection Mode** (lines 424–471):
- `--save_inspect_images` saves 5 files per sample to `./inspect/<task_id>/`: `intermediate_screenshot.png`, `ground_truth.png`, `parquet_gt_screenshot.png`, `combined_screenshot.png` (side-by-side), `raw_data.json`

### Output Format (`vl_cot_oracle_output.jsonl`)

Each line is a JSON object with fields: `id`, `sample_type` (regular/clickbait), `code_type` (web/chart), `model`, `intermediate_code`, `gt_code`, `raw_response`, `think`, `vlm_thought`, `vlm_action`, `intermediate_screenshot_b64`, `target_screenshot_b64`.

### Configuration Defaults (lines 59–84)

```python
DEFAULT_VLLM_URL = "http://localhost:7003"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
TEMPERATURE = 0.6, TOP_P = 0.95, TOP_K = 20, MAX_TOKENS = 16384
HTML_VIEWPORT_W = 1280, HTML_VIEWPORT_H = 720
MPL_DPI = 100, MPL_FIGSIZE = (10, 8)
```
