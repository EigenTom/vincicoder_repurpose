#!/usr/bin/env python3
"""
VL-CoT Synthesis Pipeline v2 — Oracle-guided, Regular + Clickbait, dual-screenshot.

Key features:
  - Model: Qwen/Qwen3.5-122B-A10B  (override with --vllm-url / --model-id)
  - Renders Intermediate Code (Regular) or GT Code (Clickbait) → Current Screenshot
    via headless Chromium (HTML) or matplotlib Agg (Python/chart).
  - Sends BOTH Current Screenshot + Target Screenshot to the model.
  - Oracle-guided Regular prompting: model uses GT code to guarantee correctness,
    but narrates <vlm_thought> as if only seeing screenshots.
  - Clickbait prompting: Current Code = GT Code, both screenshots identical,
    model must output DONE (no edits needed).
  - inspect mode: saves intermediate_screenshot.png + ground_truth.png per sample.
  - Full b64 fields included in JSONL output.

Usage:
  pip install -r requirements.txt
  playwright install chromium

  # Quick test (10 HTML + 10 chart, 20% clickbait)
  python generate_vl_cot_v2.py --vllm-url http://localhost:7001 --model-id Qwen/Qwen3.5-122B-A10B

  # Full run with Qwen
  python generate_vl_cot_v2.py --num_html 1000 --num_chart 1000 --clickbait_ratio 0.2

  # With inspect images saved
  python generate_vl_cot_v2.py --num_html 10 --num_chart 10 --save_inspect_images
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from PIL import Image

# ---------------------------------------------------------------------------
# CRITICAL: clear proxies before every API call to avoid 403 / Timeout
# ---------------------------------------------------------------------------
def _clear_proxies() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Default configuration  (overridable via CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_VLLM_URL   = "http://localhost:7003"   # Qwen port (change once loaded)
DEFAULT_MODEL_ID   = "Qwen/Qwen3.5-122B-A10B"
API_KEY            = "sk-123456"

# Qwen Thinking-mode params (official recommendation for code tasks)
TEMPERATURE        = 0.6
TOP_P              = 0.95
TOP_K              = 20
MIN_P              = 0.0
PRESENCE_PENALTY   = 0.0
REPETITION_PENALTY = 1.0
MAX_TOKENS         = 16384
TIMEOUT_SEC        = 240
MAX_API_RETRIES    = 3

_SCRIPT_DIR   = Path(__file__).resolve().parent
WEB_PARQUET   = _SCRIPT_DIR / "vincicoder_sft" / "web2html_refine.parquet"
CHART_PARQUET = _SCRIPT_DIR / "vincicoder_sft" / "chart2code_refine.parquet"
OUTPUT_DIR    = _SCRIPT_DIR
INSPECT_BASE  = _SCRIPT_DIR / "inspect"

RANDOM_SEED       = 42
# 16:9 aspect ratio
HTML_VIEWPORT_W   = 1280
HTML_VIEWPORT_H   = 720
MPL_DPI           = 100


# ===========================================================================
# Prompt Templates — Regular (Oracle-guided fix)
# ===========================================================================
SYSTEM_PROMPT_REGULAR = """\
You are an expert Data Synthesizer acting as an "Oracle". Your objective is to generate a high-quality "Vision-Language Chain-of-Thought (VL-CoT)" to train a Vision-Language Model (VLM).

You are provided with:
1. [CURRENT SCREENSHOT]: The image rendered from the flawed code.
2. [TARGET SCREENSHOT]: The perfect target image.
3. [CURRENT CODE]: The flawed intermediate code.
4. [GROUND TRUTH CODE] (ORACLE ONLY): The perfect code that generates the target image.

YOUR ROLE & CONSTRAINTS:
- **The Oracle Action:** You MUST use the [GROUND TRUTH CODE] to ensure the final code you output in `<vlm_action>` is 100% functionally correct and perfectly aligned with the target.
- **The Simulated VLM Thought:** However, the reasoning trace you generate inside `<vlm_thought>` MUST be written from the perspective of a VLM that DOES NOT have access to the [GROUND TRUTH CODE].
- **Strict Visual Grounding:** The simulated VLM can only derive its fix plan by visually comparing the [CURRENT SCREENSHOT] against the [TARGET SCREENSHOT]. Never mention variable names or hidden DOM structures in the thought process unless their absence/presence causes a visible artefact in the screenshots.\
"""

USER_PROMPT_REGULAR = """\
[CODE TYPE]: {code_type_label}

[CURRENT CODE] (Flawed):
```{lang}
{current_code}
```

[GROUND TRUTH CODE] (Oracle Reference - DO NOT mention this exists in your thought process):
{gt_code}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[TASK DEFINITION]

[IMAGE 1] = Current Screenshot — what the flawed code above renders to.
[IMAGE 2] = Target Screenshot — the perfect visual target.

Phase 1 — <think> (Your internal Oracle sandbox)
• Diff the Current Code against the Ground Truth Code to know exactly what needs fixing.
• Map these code edits to the visible differences between [IMAGE 1] and [IMAGE 2].
• Filter out any non-visual edits.

Phase 2 — <answer>
Output the final refinement reasoning using two sub-tags:

<vlm_thought>
Simulate a developer's natural, stream-of-consciousness debugging process purely based on visual evidence. The thought chain should contain the following contents:

"Looking at the target screenshot, I see X. But in our current render, it is Y..." (Observations)

"Checking the current code, I notice that the issue is caused by..." (Code Tracing)

"To fix this visual discrepancy, I need to modify..." (Action Plan)
</vlm_thought>

<vlm_action>
The COMPLETE, fully functional fixed code (based heavily on the Ground Truth Code to ensure correctness). No markdown code fences inside the action tag, no extra commentary.
</vlm_action>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[FORMAT] — Respond ONLY in the following structure:

<think>
...
</think>
<answer>
<vlm_thought>
...
</vlm_thought>
<vlm_action>
...
</vlm_action>
</answer>\
"""


# ===========================================================================
# Prompt Templates — Clickbait (already perfect, output DONE)
# ===========================================================================
SYSTEM_PROMPT_CLICKBAIT = """\
You are an expert Data Synthesizer acting as an "Oracle". Your objective is to generate a high-quality "Vision-Language Chain-of-Thought (VL-CoT)" to train a Vision-Language Model (VLM).

You are provided with:
1. [CURRENT SCREENSHOT]: The image rendered from the current code (which is already perfect).
2. [TARGET SCREENSHOT]: The perfect target image — visually identical to the current screenshot.
3. [CURRENT CODE]: The current code, which is already the Ground Truth — no changes needed.
4. [GROUND TRUTH CODE] (ORACLE ONLY): Identical to the Current Code — confirmed perfect.

YOUR ROLE & CONSTRAINTS:
- **The Oracle Confirmation:** As the Oracle, you confirm that the Current Code IS the Ground Truth. The visual gap between [IMAGE 1] and [IMAGE 2] is zero.
- **The Simulated VLM Thought:** The reasoning trace inside `<vlm_thought>` MUST be from the perspective of a VLM carefully comparing both screenshots, concluding that they are visually identical and no modification is needed.
- **Strict Output Rule:** The `<vlm_action>` MUST contain ONLY the exact word `DONE` — no code, no explanation, nothing else.\
"""

USER_PROMPT_CLICKBAIT = """\
[CODE TYPE]: {code_type_label}

[CURRENT CODE] (Already correct — this IS the Ground Truth):
```{lang}
{current_code}
```

[GROUND TRUTH CODE] (Oracle Reference — identical to Current Code above):
{gt_code}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[TASK DEFINITION]

[IMAGE 1] = Current Screenshot — rendered from the current (already perfect) code.
[IMAGE 2] = Target Screenshot — the perfect visual target.

Phase 1 — <think> (Your internal Oracle sandbox)
• Confirm that the Current Code and Ground Truth Code are identical — nothing to fix.
• Verify that [IMAGE 1] and [IMAGE 2] are visually indistinguishable (or nearly so).

Phase 2 — <answer>
Output the final reasoning using two sub-tags:

<vlm_thought>
Simulate a developer carefully comparing both screenshots. The thought process should:

"Looking at the target screenshot, I see X. In our current render, it is also X..." (Confirm visual match on every element: layout, colors, typography, structure, content)

"Checking the current code, everything aligns perfectly with the target." (Code Confirmation)

"Since the current render perfectly matches the target screenshot in all visual dimensions — layout, colors, and content — no modification is needed. I will stop here." (Conclusion)
</vlm_thought>

<vlm_action>
DONE
</vlm_action>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[FORMAT] — Respond ONLY in the following structure:

<think>
...
</think>
<answer>
<vlm_thought>
...
</vlm_thought>
<vlm_action>
DONE
</vlm_action>
</answer>\
"""


# ===========================================================================
# Regex extraction helpers
# ===========================================================================
def extract_intermediate_code(user_content: str, code_type: str) -> str | None:
    lang = "html" if code_type == "web" else "python"
    m = re.search(rf"```{lang}\s*\n([\s\S]*?)(?:\n```|$)", user_content, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_gt_code(assistant_content: str, code_type: str) -> str:
    lang = "html" if code_type == "web" else "python"
    m = re.search(rf"```{lang}\s*\n([\s\S]*?)(?:\n```|$)", assistant_content, re.IGNORECASE)
    return m.group(1).strip() if m else assistant_content.strip()


# ===========================================================================
# Rendering — HTML via Playwright headless Chromium (full-page capture)
# ===========================================================================
MAX_RENDER_RETRIES = 5        # max attempts before giving up on a sample
RENDER_RETRY_BASE_DELAY = 1.5 # seconds; multiplied by attempt index (linear backoff)

def render_full_page_html(
    html_content: str,
    width: int = HTML_VIEWPORT_W,
    height: int = HTML_VIEWPORT_H,
    goto_timeout_ms: int = 10000,
) -> bytes | None:
    """
    Render an HTML string to a full-page PNG using headless Chromium.

    The viewport is fixed at (width × height) so that every render shares
    the same baseline canvas.  Full-page=True captures the entire scrollable
    document height, so no content is cropped.

    Wait strategy per attempt:
      1. goto() with wait_until="load" (DOM + subresources)
      2. Try networkidle (short timeout) for async JS
      3. Short fixed sleep for CSS transitions / lazy images

    Retry strategy:
      Chromium can crash intermittently (SIGSEGV / signal 11) in container
      environments.  Each attempt spawns a completely fresh browser process.
      After MAX_RENDER_RETRIES failures the sample is skipped (returns None).
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        raise RuntimeError("playwright required: uv pip install playwright && playwright install chromium")

    tf_path: str | None = None

    for attempt in range(1, MAX_RENDER_RETRIES + 1):
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                # Fixed viewport — same for every render call
                page = browser.new_page(viewport={"width": width, "height": height})

                with tempfile.NamedTemporaryFile(
                    suffix=".html", mode="w", encoding="utf-8", delete=False
                ) as tf:
                    tf.write(html_content)
                    tf_path = tf.name

                try:
                    # Stage 1: wait for DOM + synchronous resources
                    page.goto(
                        f"file://{tf_path}",
                        timeout=goto_timeout_ms,
                        wait_until="load",
                    )
                    # Stage 2: let async JS settle (best-effort, ignore timeout)
                    try:
                        page.wait_for_load_state("networkidle", timeout=3000)
                    except PWTimeout:
                        pass
                    # Stage 3: fixed pause for CSS transitions / lazy rendering
                    page.wait_for_timeout(500)

                    # Capture the full scrollable document, not just the viewport
                    png_bytes = page.screenshot(type="png", full_page=True)
                finally:
                    try:
                        os.unlink(tf_path)
                    except OSError:
                        pass
                    tf_path = None

                browser.close()
            return png_bytes  # success

        except Exception as exc:
            # Clean up temp file if it wasn't removed inside the try block
            if tf_path:
                try:
                    os.unlink(tf_path)
                except OSError:
                    pass
                tf_path = None

            if attempt >= MAX_RENDER_RETRIES:
                logging.warning(
                    "HTML full-page rendering failed after %d attempts — skipping. Last error: %s",
                    MAX_RENDER_RETRIES, exc,
                )
                return None

            delay = RENDER_RETRY_BASE_DELAY * attempt
            logging.warning(
                "HTML render attempt %d/%d failed (%s). Retrying in %.1fs…",
                attempt, MAX_RENDER_RETRIES, type(exc).__name__, delay,
            )
            time.sleep(delay)

    return None  # unreachable, but satisfies type checker


# ===========================================================================
# Rendering — Python/Matplotlib via Agg backend (forced size)
# ===========================================================================
# Target output dimensions for every chart render
MPL_FIGSIZE = (10, 8)   # inches — overrides whatever the code sets

def render_matplotlib_chart(python_code: str) -> bytes | None:
    """
    Execute matplotlib Python code and capture the last figure as PNG.

    Consistency guarantees:
      • rcParams figsize and dpi are forced BEFORE exec so the figure
        created by plt.figure() / plt.subplots() uses our defaults.
      • After exec, ALL figures are forcibly resized to MPL_FIGSIZE so that
        even explicit figsize= calls in user code are overridden.
      • Saved with dpi=MPL_DPI and bbox_inches=None for exact pixel size.
        Result: every chart image is exactly
        (MPL_FIGSIZE[0]*MPL_DPI) × (MPL_FIGSIZE[1]*MPL_DPI) pixels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Force default size before exec
    plt.rcParams["figure.figsize"] = list(MPL_FIGSIZE)
    plt.rcParams["figure.dpi"]     = MPL_DPI
    plt.close("all")

    ns: dict[str, Any] = {}
    try:
        exec(python_code, ns)  # noqa: S102
    except SystemExit:
        pass
    except Exception as exc:
        logging.warning("matplotlib exec error: %s", exc)
        # Fall through — still try to capture partial figures

    fig_nums = plt.get_fignums()
    if not fig_nums:
        logging.warning("matplotlib exec produced no figures")
        return None

    # Force-resize every figure to the canonical size AFTER exec
    for fn in fig_nums:
        try:
            plt.figure(fn).set_size_inches(*MPL_FIGSIZE, forward=True)
        except Exception:
            pass

    buf = io.BytesIO()
    try:
        fig = plt.figure(fig_nums[-1])
        # bbox_inches=None → exact MPL_FIGSIZE × MPL_DPI pixel output
        fig.savefig(buf, format="png", dpi=MPL_DPI, bbox_inches=None)
    except Exception as exc:
        logging.warning("matplotlib savefig failed: %s", exc)
        return None
    finally:
        plt.close("all")

    buf.seek(0)
    return buf.read()


def render_code(code: str, code_type: str) -> bytes | None:
    """Dispatch to the appropriate renderer.  Both return PNG bytes."""
    if code_type == "web":
        return render_full_page_html(code)
    else:
        return render_matplotlib_chart(code)


# ===========================================================================
# Inspect mode — save PNG files + raw data JSON for human review
# ===========================================================================
def save_inspect_images(
    task_id: str,
    intermediate_png: bytes,
    target_png: bytes,
    parquet_image_bytes: bytes,
    user_content: str,
    assistant_content: str,
    intermediate_code: str,
    gt_code: str,
    sample_type: str,
) -> None:
    out_dir = INSPECT_BASE / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Rendered screenshots ────────────────────────────────────────────────
    (out_dir / "intermediate_screenshot.png").write_bytes(intermediate_png)
    (out_dir / "ground_truth.png").write_bytes(target_png)

    # ── Raw parquet image (original dataset screenshot, for cross-checking) ─
    # Convert to PNG regardless of original format (jpeg / png / etc.)
    try:
        parquet_img = Image.open(io.BytesIO(parquet_image_bytes)).convert("RGB")
        parquet_img.save(out_dir / "parquet_gt_screenshot.png", format="PNG")
    except Exception as exc:
        logging.warning("[inspect][%s] could not save parquet image: %s", task_id, exc)

    # ── Side-by-side: intermediate | gap | rendered GT ─────────────────────
    intermediate_img = Image.open(io.BytesIO(intermediate_png))
    target_img       = Image.open(io.BytesIO(target_png))
    combined_width   = intermediate_img.width + 20 + target_img.width
    combined_height  = max(intermediate_img.height, target_img.height)
    combined_png     = Image.new("RGB", (combined_width, combined_height), color=(200, 200, 200))
    combined_png.paste(intermediate_img.convert("RGB"), (0, 0))
    combined_png.paste(target_img.convert("RGB"), (intermediate_img.width + 20, 0))
    combined_png.save(out_dir / "combined_screenshot.png", format="PNG")

    # ── Raw data JSON for code-level inspection ─────────────────────────────
    raw_data = {
        "task_id":                    task_id,
        "sample_type":                sample_type,
        "parquet_user_content":       user_content,
        "parquet_assistant_content":  assistant_content,
        "extracted_intermediate_code": intermediate_code,
        "extracted_gt_code":          gt_code,
    }
    (out_dir / "raw_data.json").write_text(
        json.dumps(raw_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ===========================================================================
# API client — multimodal dual-image call
# ===========================================================================
def call_vllm_dual_image(
    system: str,
    text: str,
    current_b64: str,
    target_b64: str,
    current_fmt: str,
    target_fmt: str,
    vllm_url: str,
    model_id: str,
) -> str | None:
    """
    POST multimodal request: [text] → [IMAGE 1: current] → [IMAGE 2: target].
    Proxies are cleared before each attempt (hard rule).
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx required: uv pip install httpx")

    url = f"{vllm_url.rstrip('/')}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/{current_fmt};base64,{current_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/{target_fmt};base64,{target_b64}"}},
                ],
            },
        ],
        "max_tokens":          MAX_TOKENS,
        "temperature":         TEMPERATURE,
        "top_p":               TOP_P,
        "top_k":               TOP_K,
        "min_p":               MIN_P,
        "presence_penalty":    PRESENCE_PENALTY,
        "repetition_penalty":  REPETITION_PENALTY,
    }

    for attempt in range(MAX_API_RETRIES + 1):
        _clear_proxies()  # CRITICAL: unset proxies before every request
        try:
            with httpx.Client(timeout=TIMEOUT_SEC) as client:
                resp = client.post(
                    url, json=payload,
                    headers={"Authorization": f"Bearer {API_KEY}"},
                )
                resp.raise_for_status()
                data = resp.json()
            choice  = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "").strip()
            return content or None
        except Exception as exc:
            logging.warning("API attempt %d/%d failed: %s", attempt + 1, MAX_API_RETRIES + 1, exc)
            if attempt < MAX_API_RETRIES:
                time.sleep(2.0 * (attempt + 1))
    return None


# ===========================================================================
# Response parser (handles models that merge <think> into <vlm_thought>)
# ===========================================================================
def parse_response(raw: str) -> tuple[str, str, str]:
    """
    Return (think, vlm_thought, vlm_action) from raw model response.

    Handles three edge-case formats:
      1. Standard:  <think>…</think> … <vlm_thought>…</vlm_thought> <vlm_action>…</vlm_action>
      2. Collapsed: <think>…</vlm_thought> (model merged both, no opening <vlm_thought>)
      3. Truncated: <vlm_thought>… (response cut off before closing tag — extract to end)
    """
    think_m       = re.search(r"<think>([\s\S]*?)</think>",             raw)
    vlm_thought_m = re.search(r"<vlm_thought>([\s\S]*?)</vlm_thought>", raw)
    vlm_action_m  = re.search(r"<vlm_action>([\s\S]*?)</vlm_action>",   raw)

    # ── think ──────────────────────────────────────────────────────────────
    combined_think = ""
    if think_m:
        combined_think = think_m.group(1).strip()
    elif re.search(r"</vlm_thought>", raw):
        # Collapsed: <think> content runs up to </vlm_thought>
        orphan = re.search(r"<think>([\s\S]*?)</vlm_thought>", raw)
        if orphan:
            combined_think = orphan.group(1).strip()

    # ── vlm_thought ─────────────────────────────────────────────────────────
    if vlm_thought_m:
        vlm_thought = vlm_thought_m.group(1).strip()
    else:
        # Truncated response: <vlm_thought> opened but never closed
        trunc = re.search(r"<vlm_thought>([\s\S]+)$", raw)
        vlm_thought = trunc.group(1).strip() if trunc else combined_think

    # ── vlm_action ──────────────────────────────────────────────────────────
    vlm_action = vlm_action_m.group(1).strip() if vlm_action_m else ""

    return combined_think, vlm_thought, vlm_action


# ===========================================================================
# Per-sample processing — Regular
# ===========================================================================
def process_regular(
    row: Any,
    task_id: str,
    code_type: str,
    vllm_url: str,
    model_id: str,
    save_inspect: bool,
) -> dict[str, Any] | None:
    lang          = "html"           if code_type == "web" else "python"
    code_type_lbl = "Web (HTML/CSS)" if code_type == "web" else "Chart (Python/Matplotlib)"

    intermediate_code = extract_intermediate_code(row["user_content"], code_type)
    if not intermediate_code:
        logging.warning("[Regular][%s] %s: intermediate code extraction failed", code_type, task_id)
        return None
    gt_code = extract_gt_code(row["assistant_content"], code_type)

    # Render flawed intermediate code → [IMAGE 1] current screenshot
    t0 = time.time()
    current_png = render_code(intermediate_code, code_type)
    if current_png is None:
        logging.warning("[Regular][%s] %s: intermediate render failed (%.1fs)", code_type, task_id, time.time() - t0)
        return None
    logging.info("[Regular][%s] %s  intermediate render=%.1fs  bytes=%d",
                 code_type, task_id, time.time() - t0, len(current_png))

    # Render GT code with IDENTICAL browser/renderer params → [IMAGE 2] target screenshot
    # (Discards the raw parquet image to guarantee 1:1 resolution/viewport consistency.)
    t1 = time.time()
    target_png = render_code(gt_code, code_type)
    if target_png is None:
        logging.warning("[Regular][%s] %s: GT render failed (%.1fs)", code_type, task_id, time.time() - t1)
        return None
    logging.info("[Regular][%s] %s  GT render=%.1fs  bytes=%d",
                 code_type, task_id, time.time() - t1, len(target_png))

    current_b64 = base64.b64encode(current_png).decode()
    target_b64  = base64.b64encode(target_png).decode()

    if save_inspect:
        save_inspect_images(
            task_id=task_id,
            intermediate_png=current_png,
            target_png=target_png,
            parquet_image_bytes=row["image"],
            user_content=row["user_content"],
            assistant_content=row["assistant_content"],
            intermediate_code=intermediate_code,
            gt_code=gt_code,
            sample_type="regular",
        )

    user_prompt = USER_PROMPT_REGULAR.format(
        code_type_label=code_type_lbl, lang=lang,
        current_code=intermediate_code, gt_code=gt_code,
    )

    t2  = time.time()
    raw = call_vllm_dual_image(
        system=SYSTEM_PROMPT_REGULAR, text=user_prompt,
        current_b64=current_b64, target_b64=target_b64,
        current_fmt="png", target_fmt="png",
        vllm_url=vllm_url, model_id=model_id,
    )
    if not raw:
        logging.warning("[Regular][%s] %s: API returned None", code_type, task_id)
        return None
    logging.info("[Regular][%s] %s ✓  resp=%d  api=%.1fs", code_type, task_id, len(raw), time.time() - t2)

    think, vlm_thought, vlm_action = parse_response(raw)
    return {
        "id":                       task_id,
        "sample_type":              "regular",
        "code_type":                code_type,
        "model":                    model_id,
        "intermediate_code":        intermediate_code,
        "gt_code":                  gt_code,
        "raw_response":             raw,
        "think":                    think,
        "vlm_thought":              vlm_thought,
        "vlm_action":               vlm_action,
        "intermediate_screenshot_b64": current_b64,
        "target_screenshot_b64":       target_b64,
    }


# ===========================================================================
# Per-sample processing — Clickbait
# ===========================================================================
def process_clickbait(
    row: Any,
    task_id: str,
    code_type: str,
    vllm_url: str,
    model_id: str,
    save_inspect: bool,
) -> dict[str, Any] | None:
    lang          = "html"           if code_type == "web" else "python"
    code_type_lbl = "Web (HTML/CSS)" if code_type == "web" else "Chart (Python/Matplotlib)"

    intermediate_code = extract_intermediate_code(row["user_content"], code_type)
    if not intermediate_code:
        logging.warning("[Clickbait][%s] %s: intermediate code extraction failed", code_type, task_id)
        return None
    gt_code = extract_gt_code(row["assistant_content"], code_type)

    # For clickbait, current_code = GT code → both images rendered identically.
    # Render GT code once; reuse the same PNG bytes for both [IMAGE 1] and [IMAGE 2].
    # This guarantees pixel-perfect identity so the model correctly sees no gap.
    t0 = time.time()
    current_png = render_code(gt_code, code_type)
    if current_png is None:
        logging.warning("[Clickbait][%s] %s: GT code rendering failed (%.1fs)", code_type, task_id, time.time() - t0)
        return None
    logging.info("[Clickbait][%s] %s  GT render=%.1fs  bytes=%d",
                 code_type, task_id, time.time() - t0, len(current_png))

    # Both images are the same render — viewer and target are visually identical
    target_png  = current_png
    current_b64 = base64.b64encode(current_png).decode()
    target_b64  = current_b64   # identical bytes → identical b64

    if save_inspect:
        save_inspect_images(
            task_id=task_id,
            intermediate_png=current_png,
            target_png=target_png,
            parquet_image_bytes=row["image"],
            user_content=row["user_content"],
            assistant_content=row["assistant_content"],
            intermediate_code=intermediate_code,
            gt_code=gt_code,
            sample_type="clickbait",
        )

    user_prompt = USER_PROMPT_CLICKBAIT.format(
        code_type_label=code_type_lbl, lang=lang,
        current_code=gt_code, gt_code=gt_code,
    )

    t1  = time.time()
    raw = call_vllm_dual_image(
        system=SYSTEM_PROMPT_CLICKBAIT, text=user_prompt,
        current_b64=current_b64, target_b64=target_b64,
        current_fmt="png", target_fmt="png",
        vllm_url=vllm_url, model_id=model_id,
    )
    if not raw:
        logging.warning("[Clickbait][%s] %s: API returned None", code_type, task_id)
        return None
    logging.info("[Clickbait][%s] %s ✓  resp=%d  api=%.1fs", code_type, task_id, len(raw), time.time() - t1)

    think, vlm_thought, vlm_action = parse_response(raw)

    # Validate: vlm_action should be exactly "DONE"
    if vlm_action.strip().upper() != "DONE":
        logging.warning(
            "[Clickbait][%s] %s: vlm_action is not DONE (got %r) — keeping but flagging",
            code_type, task_id, vlm_action[:40]
        )

    return {
        "id":                       task_id,
        "sample_type":              "clickbait",
        "code_type":                code_type,
        "model":                    model_id,
        "intermediate_code":        intermediate_code,   # original flawed code (for reference)
        "gt_code":                  gt_code,
        "raw_response":             raw,
        "think":                    think,
        "vlm_thought":              vlm_thought,
        "vlm_action":               vlm_action,
        "intermediate_screenshot_b64": current_b64,      # render of GT (looks like target)
        "target_screenshot_b64":       target_b64,
    }


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="VL-CoT Oracle-guided synthesis pipeline v2 (Regular + Clickbait)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vllm-url",          default=DEFAULT_VLLM_URL,  help="vLLM server base URL")
    parser.add_argument("--model-id",          default=DEFAULT_MODEL_ID,  help="Model ID served by vLLM")
    parser.add_argument("--num_html",          type=int,   default=10,    help="# HTML/Web samples to synthesise (-1 = all)")
    parser.add_argument("--num_chart",         type=int,   default=10,    help="# Chart samples to synthesise (-1 = all)")
    parser.add_argument("--clickbait_ratio",   type=float, default=0.2,   help="Fraction of samples that are Clickbait [0,1]")
    parser.add_argument("--save_inspect_images", action="store_true",     help="Save intermediate/GT PNGs for human review")
    parser.add_argument("--seed",              type=int,   default=RANDOM_SEED)
    parser.add_argument("--output",            default=str(OUTPUT_DIR / "vl_cot_oracle_output.jsonl"))
    parser.add_argument("--render-test-only",  action="store_true",       help="Run render smoke-test only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("httpx").setLevel(logging.CRITICAL)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):  # type: ignore[misc]
            return it

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert 0.0 <= args.clickbait_ratio <= 1.0, "--clickbait_ratio must be in [0, 1]"

    logging.info("=" * 70)
    logging.info("Model  : %s", args.model_id)
    logging.info("Server : %s", args.vllm_url)
    logging.info("num_html=%d  num_chart=%d  clickbait=%.0f%%  inspect=%s",
                 args.num_html, args.num_chart, args.clickbait_ratio * 100,
                 args.save_inspect_images)
    logging.info("=" * 70)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    import pandas as pd

    print("\n[DATA LOADING]")
    df_web = pd.read_parquet(WEB_PARQUET)
    print(f"  web2html_refine.parquet  → {len(df_web):,} rows  columns={list(df_web.columns)}")

    df_chart = pd.read_parquet(CHART_PARQUET)
    print(f"  chart2code_refine.parquet → {len(df_chart):,} rows  columns={list(df_chart.columns)}")
    print()

    # -----------------------------------------------------------------------
    # Sample indices
    # -----------------------------------------------------------------------
    def sample_indices(df: pd.DataFrame, n: int, rng: random.Random) -> list[int]:
        total = len(df)
        if n == -1 or n >= total:
            return list(range(total))
        return rng.sample(range(total), n)

    web_indices   = sample_indices(df_web,   args.num_html,  rng)
    chart_indices = sample_indices(df_chart, args.num_chart, rng)

    logging.info("Sampled  html=%d  chart=%d", len(web_indices), len(chart_indices))

    # -----------------------------------------------------------------------
    # Assign Regular / Clickbait split per modality
    # -----------------------------------------------------------------------
    def split_clickbait(indices: list[int], ratio: float, rng: random.Random) -> tuple[list[int], list[int]]:
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_bait = max(0, round(len(shuffled) * ratio))
        return shuffled[n_bait:], shuffled[:n_bait]   # (regular, clickbait)

    web_regular,   web_clickbait   = split_clickbait(web_indices,   args.clickbait_ratio, rng)
    chart_regular, chart_clickbait = split_clickbait(chart_indices, args.clickbait_ratio, rng)

    print("[SPLIT SUMMARY]")
    print(f"  HTML  : {len(web_regular)} regular  +  {len(web_clickbait)} clickbait")
    print(f"  Chart : {len(chart_regular)} regular  +  {len(chart_clickbait)} clickbait")
    print()

    # -----------------------------------------------------------------------
    # Render smoke-test (first 2 of each type/group, no API)
    # -----------------------------------------------------------------------
    print("[RENDER SMOKE TEST]")
    for label, df, indices, code_type in [
        ("html-regular",    df_web,   web_regular[:2],   "web"),
        ("chart-regular",   df_chart, chart_regular[:2], "chart"),
    ]:
        for idx in indices:
            code = extract_intermediate_code(df.iloc[idx]["user_content"], code_type)
            png  = render_code(code, code_type) if code else None
            ok   = "✓" if png else "✗ FAIL"
            print(f"  [{label}] df_idx={idx}  {ok}  bytes={len(png) if png else 0}")
    print()

    if args.render_test_only:
        logging.info("--render-test-only set — exiting after render test.")
        return

    # -----------------------------------------------------------------------
    # Main synthesis loop
    # -----------------------------------------------------------------------
    results: list[dict[str, Any]] = []
    total_samples = (len(web_regular) + len(web_clickbait) +
                     len(chart_regular) + len(chart_clickbait))

    # Counter for sequential IDs per code_type
    html_seq  = 0
    chart_seq = 0

    def next_task_id(code_type: str, sample_type: str) -> str:
        nonlocal html_seq, chart_seq
        prefix = "html" if code_type == "web" else "chart"
        if code_type == "web":
            html_seq += 1
            seq = html_seq
        else:
            chart_seq += 1
            seq = chart_seq
        suffix = "_cb" if sample_type == "clickbait" else ""
        return f"{prefix}{suffix}_{seq:04d}"

    batches = [
        ("HTML  Regular",   df_web,   web_regular,    "web",   "regular"),
        ("HTML  Clickbait", df_web,   web_clickbait,  "web",   "clickbait"),
        ("Chart Regular",   df_chart, chart_regular,  "chart", "regular"),
        ("Chart Clickbait", df_chart, chart_clickbait,"chart", "clickbait"),
    ]

    for batch_label, df, indices, code_type, sample_type in batches:
        if not indices:
            continue
        print(f"\n{'─'*60}")
        print(f"  Processing: {batch_label}  ({len(indices)} samples)")
        print(f"{'─'*60}")

        for idx in tqdm(indices, desc=batch_label, unit="sample"):
            task_id = next_task_id(code_type, sample_type)
            row = df.iloc[idx]

            if sample_type == "regular":
                result = process_regular(
                    row=row, task_id=task_id, code_type=code_type,
                    vllm_url=args.vllm_url, model_id=args.model_id,
                    save_inspect=args.save_inspect_images,
                )
            else:
                result = process_clickbait(
                    row=row, task_id=task_id, code_type=code_type,
                    vllm_url=args.vllm_url, model_id=args.model_id,
                    save_inspect=args.save_inspect_images,
                )

            if result:
                results.append(result)
            else:
                logging.warning("[%s] %s skipped", batch_label, task_id)

    # -----------------------------------------------------------------------
    # Save JSONL (all fields including b64 images)
    # -----------------------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    reg_ok = sum(1 for r in results if r["sample_type"] == "regular")
    bait_ok = sum(1 for r in results if r["sample_type"] == "clickbait")
    bait_done = sum(1 for r in results if r["sample_type"] == "clickbait"
                    and r["vlm_action"].strip().upper() == "DONE")
    web_ok   = sum(1 for r in results if r["code_type"] == "web")
    chart_ok = sum(1 for r in results if r["code_type"] == "chart")

    print("\n" + "=" * 70)
    print(f"DONE — saved {len(results)} / {total_samples} samples")
    print(f"  HTML  : {web_ok}   Chart: {chart_ok}")
    print(f"  Regular  : {reg_ok}   Clickbait: {bait_ok}  (DONE-correct: {bait_done}/{bait_ok})")
    print(f"  Output   : {output_path}")
    if args.save_inspect_images:
        print(f"  Inspect  : {INSPECT_BASE}/")
    print()
    print("PAUSING — please inspect the .jsonl file to verify CoT quality")
    print("before proceeding to the full run.")
    print("=" * 70)

if __name__ == "__main__":
    main()
