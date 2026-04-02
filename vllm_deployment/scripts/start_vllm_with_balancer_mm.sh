#!/bin/bash
# Multimodal vLLM deployment: Qwen3.5-122B-A10B (MoE) + Load Balancer
# Usage: bash ./scripts/start_vllm_with_balancer_mm.sh

set -e

# ---------------------------------------------------------------------------
# Proxy isolation: corporate proxy must NOT intercept loopback traffic.
# ---------------------------------------------------------------------------
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# ---------------------------------------------------------------------------
# Environment: model cache & HuggingFace paths (keep off host main disk)
# ---------------------------------------------------------------------------
# export HF_HOME="/map-vepfs/huggingface/"
# export HF_HUB_CACHE="/map-vepfs/huggingface/hub"
# export HF_ENDPOINT="https://hf-mirror.com"
# export VLLM_CACHE_ROOT="/map-vepfs/yi/vl_coding/vllm_deployment/vllm_cache"
# export TRITON_CACHE_DIR="/map-vepfs/yi/vl_coding/vllm_deployment/triton_cache"

# ---------------------------------------------------------------------------
# Deployment paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# VENV_ROOT="/map-vepfs/yi/vl_coding/.venv"   # project virtual-env
VENV_ROOT="$DEPLOY_ROOT/.venv"
VLLM_BIN="$VENV_ROOT/bin/vllm"
cd "$DEPLOY_ROOT"

# Verify vllm binary exists
if [ ! -x "$VLLM_BIN" ]; then
    echo "ERROR: vllm binary not found at $VLLM_BIN"
    exit 1
fi
echo "Using vllm: $VLLM_BIN ($(\"$VLLM_BIN\" --version 2>/dev/null || echo 'version unknown'))"

# ---------------------------------------------------------------------------
# Model & GPU configuration
# ---------------------------------------------------------------------------
MODEL_NAME="Qwen/Qwen3.5-122B-A10B-FP8"

# DP + Expert-Parallel (EP) is the recommended strategy for MoE models.
# All 8 A800-80GB GPUs are used as a single server with DP=8 and EP sharding.
# Each GPU holds ~1/8 of the MoE expert weights (~15 GB) plus replicated dense
# layers, leaving ample headroom for KV cache on 80 GB cards.
#
# GPU_CONFIG format: "<comma-separated GPU IDs>:<listen port>"
# Use semicolons to separate multiple workers (not needed here with full DP).
GPU_CONFIG="2,3:7001"

OVERALL_TP_SIZE=2
DP_SIZE=1                   # --data-parallel-size: one replica per GPU
GPU_MEMORY_UTILIZATION=0.99 # slightly reduced to give EP headroom
MAX_MODEL_LEN=131072         # 128 k tokens: images + long thinking chain
# MAX_MODEL_LEN=2048

# ---------------------------------------------------------------------------
# Load balancer configuration
# ---------------------------------------------------------------------------
LB_PORT=7000
LB_STRATEGY="round_robin"
LB_HEALTH_CHECK_INTERVAL=10.0

# ---------------------------------------------------------------------------
# Log & config directories
# ---------------------------------------------------------------------------
LOG_DIR="$DEPLOY_ROOT/logs/vllm"
ENDPOINT_POOL_FILE="$DEPLOY_ROOT/configs/vllm_endpoint_pool.txt"
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$ENDPOINT_POOL_FILE")"

echo "=========================================="
echo "Starting Multimodal vLLM + Load Balancer"
echo "=========================================="
echo "Deployment root : $DEPLOY_ROOT"
echo "Model           : $MODEL_NAME"
echo "GPU Config      : $GPU_CONFIG"
echo "DP size         : $DP_SIZE"
echo "Load Balancer   : port $LB_PORT"
echo "Log Directory   : $LOG_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Start vLLM workers
# ---------------------------------------------------------------------------
echo "=== Step 1: Starting vLLM workers ==="
echo ""

> "$ENDPOINT_POOL_FILE"

IFS=';' read -ra GPU_CONFIGS <<< "$GPU_CONFIG"

VLLM_PIDS=()

for worker_cfg in "${GPU_CONFIGS[@]}"; do
    gpu_ids="${worker_cfg%:*}"
    port="${worker_cfg##*:}"
    gpu_ids="$(echo "$gpu_ids" | tr -d ' ')"
    port="$(echo "$port" | tr -d ' ')"

    # Derive TP from DP: with full DP, TP=1 (EP handles the sharding).
    TP_SIZE=$OVERALL_TP_SIZE

    echo "Starting vLLM on GPU(s) [$gpu_ids], port $port, DP=$DP_SIZE, TP=$TP_SIZE ..."

    LOG_FILE="$LOG_DIR/vllm_gpu${gpu_ids/,/_}_port${port}.log"

    (
        # ---- subshell: full env isolation (for map) ----
        # export HF_HOME="/map-vepfs/huggingface/"
        # export HF_HUB_CACHE="/map-vepfs/huggingface/hub"
        # export HF_ENDPOINT="https://hf-mirror.com"
        # export VLLM_CACHE_ROOT="/map-vepfs/yi/vl_coding/vllm_deployment/vllm_cache"
        # export TRITON_CACHE_DIR="/map-vepfs/yi/vl_coding/vllm_deployment/triton_cache"
        # unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

        export CUDA_VISIBLE_DEVICES="$gpu_ids"

        # Activate venv so all Python deps are on PATH
        source "$VENV_ROOT/bin/activate"

        echo "=== GPU(s) $gpu_ids, Port $port ===" >  "$LOG_FILE"
        echo "Starting at $(date)"                  >> "$LOG_FILE"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"

        "$VLLM_BIN" serve "$MODEL_NAME" \
            --port "$port" \
            -dp "$DP_SIZE" \
            --tensor-parallel-size "$TP_SIZE" \
            --enable-expert-parallel \
            --mm-encoder-tp-mode data \
            --mm-processor-cache-type shm \
            --reasoning-parser qwen3 \
            --enable-prefix-caching \
            --limit-mm-per-prompt '{"image": 2}' \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --trust-remote-code \
            --dtype bfloat16 \
            >> "$LOG_FILE" 2>&1
    ) &

    PID=$!
    VLLM_PIDS+=($PID)

    echo "http://localhost:$port/v1" >> "$ENDPOINT_POOL_FILE"

    echo "  -> Started with PID $PID"
    echo "  -> Endpoint : http://localhost:$port/v1"
    echo "  -> Log      : $LOG_FILE"

    sleep 5
done

printf "%s\n" "${VLLM_PIDS[@]}" > "$LOG_DIR/vllm_pids.txt"
echo ""
echo "vLLM PID(s) saved to: $LOG_DIR/vllm_pids.txt"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Wait for workers to finish loading
# ---------------------------------------------------------------------------
echo "=== Step 2: Waiting for model load (122B MoE, ~5-10 min) ==="
echo "Polling every 30 s for up to 15 minutes..."
echo ""

MAX_WAIT=900   # 15 minutes
POLL_INTERVAL=30
ELAPSED=0
ALL_HEALTHY=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    HEALTHY_COUNT=0
    TOTAL_COUNT=0
    for worker_cfg in "${GPU_CONFIGS[@]}"; do
        port="${worker_cfg##*:}"
        port="$(echo "$port" | tr -d ' ')"
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        if curl -sf --noproxy '*' --max-time 5 "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
        fi
    done

    if [ "$HEALTHY_COUNT" -eq "$TOTAL_COUNT" ]; then
        echo "All $TOTAL_COUNT worker(s) healthy after ${ELAPSED}s."
        ALL_HEALTHY=1
        break
    fi

    echo "  [${ELAPSED}s] $HEALTHY_COUNT/$TOTAL_COUNT ready — still waiting..."
    sleep $POLL_INTERVAL
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ $ALL_HEALTHY -eq 0 ]; then
    echo ""
    echo "WARNING: Not all workers became healthy within ${MAX_WAIT}s."
    echo "They may still be loading. Check logs in $LOG_DIR/"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 3: Start load balancer
# ---------------------------------------------------------------------------
echo "=== Step 3: Starting Load Balancer ==="
echo ""

BACKEND_URLS=()
for worker_cfg in "${GPU_CONFIGS[@]}"; do
    port="${worker_cfg##*:}"
    port="$(echo "$port" | tr -d ' ')"
    BACKEND_URLS+=("http://localhost:$port/v1")
done

echo "Load Balancer Configuration:"
echo "  Port     : $LB_PORT"
echo "  Strategy : $LB_STRATEGY"
echo "  Backends : ${BACKEND_URLS[*]}"
echo ""

# Activate venv for load balancer Python process too
source "$VENV_ROOT/bin/activate"

python3 -c "import fastapi" 2>/dev/null || {
    echo "Error: FastAPI not installed in $VENV_ROOT. Run: $VENV_ROOT/bin/pip install fastapi uvicorn httpx"
    exit 1
}

echo "Starting load balancer..."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
nohup python3 "$DEPLOY_ROOT/load_balancer.py" \
    --backends "${BACKEND_URLS[@]}" \
    --host 0.0.0.0 \
    --port "$LB_PORT" \
    --strategy "$LB_STRATEGY" \
    --health-check-interval "$LB_HEALTH_CHECK_INTERVAL" \
    > "$LOG_DIR/load_balancer_port${LB_PORT}.log" 2>&1 &

LB_PID=$!
echo "$LB_PID" > "$LOG_DIR/vllm_lb_pid.txt"

echo "  -> Load balancer started with PID $LB_PID"
echo "  -> Endpoint : http://localhost:$LB_PORT"
echo "  -> Log      : $LOG_DIR/load_balancer_port${LB_PORT}.log"

sleep 5

echo ""
echo "Checking load balancer health..."
if curl -sf --noproxy '*' --max-time 5 "http://localhost:$LB_PORT/health" > /dev/null 2>&1; then
    echo "  Load balancer: HEALTHY"
    curl -s --noproxy '*' "http://localhost:$LB_PORT/health" | python3 -m json.tool 2>/dev/null \
        || curl -s --noproxy '*' "http://localhost:$LB_PORT/health"
else
    echo "  Load balancer: NOT READY — check $LOG_DIR/load_balancer_port${LB_PORT}.log"
fi

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "vLLM Worker(s):"
for i in "${!GPU_CONFIGS[@]}"; do
    worker_cfg="${GPU_CONFIGS[$i]}"
    gpu_ids="${worker_cfg%:*}"
    port="${worker_cfg##*:}"
    port="$(echo "$port" | tr -d ' ')"
    PID="${VLLM_PIDS[$i]}"
    echo "  GPU(s) $gpu_ids → http://localhost:$port/v1  (PID: $PID)"
done
echo ""
echo "Load Balancer:"
echo "  http://localhost:$LB_PORT  (PID: $LB_PID)"
echo ""
echo "Client config:"
echo "  base_url = \"http://localhost:$LB_PORT/v1\""
echo "  Endpoint pool: $ENDPOINT_POOL_FILE"
echo ""
echo "To stop all services:"
echo "  $SCRIPT_DIR/stop_vllm_services.sh"
echo ""
