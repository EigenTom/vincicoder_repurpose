#!/bin/bash
# Start Python Load Balancer for vLLM backends (standalone deployment)
# Usage: ./scripts/start_load_balancer.sh [num_instances] [base_port] [lb_port]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DEPLOY_ROOT"

NUM_INSTANCES="${1:-4}"
BASE_PORT="${2:-8000}"
LB_PORT="${3:-$BASE_PORT}"

echo "Starting Load Balancer for vLLM"
echo "Number of instances: $NUM_INSTANCES"
echo "Base port: $BASE_PORT"
echo "Load balancer port: $LB_PORT"
echo ""

# Activate virtual environment if it exists
if [ -d "$DEPLOY_ROOT/.venv" ]; then
    source "$DEPLOY_ROOT/.venv/bin/activate"
elif [ -d "$DEPLOY_ROOT/../.venv" ]; then
    source "$DEPLOY_ROOT/../.venv/bin/activate"
fi

# Check if FastAPI is installed
python3 -c "import fastapi" 2>/dev/null || {
    echo "Error: FastAPI not installed. Install with: pip install fastapi uvicorn httpx"
    exit 1
}

# Build backend list (vLLM style: /v1 suffix)
BACKENDS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    BACKENDS+=("http://localhost:${PORT}/v1")
done

LB_LOG_DIR="$DEPLOY_ROOT/logs/vllm"
mkdir -p "$LB_LOG_DIR"

echo "Backends:"
for backend in "${BACKENDS[@]}"; do
    echo "  - $backend"
done
echo ""

echo "Starting load balancer..."
nohup python3 "$DEPLOY_ROOT/load_balancer.py" \
    --backends "${BACKENDS[@]}" \
    --host 0.0.0.0 \
    --port "$LB_PORT" \
    --strategy round_robin \
    --health-check-interval 10.0 \
    > "${LB_LOG_DIR}/load_balancer_vllm_port${LB_PORT}.log" 2>&1 &

LB_PID=$!

PID_FILE="$LB_LOG_DIR/vllm_lb_pid.txt"
echo "$LB_PID" > "$PID_FILE"

echo "Load balancer started with PID: $LB_PID"
echo "Load balancer URL: http://localhost:${LB_PORT}"
echo "PID saved to: $PID_FILE"
echo ""
echo "To check status: curl http://localhost:${LB_PORT}/health"
echo "To stop: $SCRIPT_DIR/stop_vllm_services.sh"
echo "Or manually: kill $LB_PID"
