# vLLM Deployment (Load Balancer + vLLM)

Standalone folder for running vLLM instances behind a Python load balancer. No dependency on ReviewGrounder or other projects.

## Layout

```
vllm_deployment/
├── load_balancer.py       # FastAPI load balancer (round_robin / least_conn)
├── scripts/
│   ├── start_vllm_with_balancer.sh   # Start vLLM on multiple GPUs + LB
│   ├── stop_vllm_services.sh         # Stop vLLM and LB (only PIDs we started)
│   └── start_load_balancer.sh         # Start LB only (existing backends)
├── configs/
│   └── vllm_endpoint_pool.txt        # Written at start; one URL per line
├── logs/
│   └── vllm/                         # vLLM and LB logs, PIDs
├── requirements.txt
└── README.md
```

## Requirements

- **vLLM**: installed and on `PATH` (e.g. `pip install vllm` or project venv).
- **Python deps for LB**: `pip install -r requirements.txt` (fastapi, uvicorn, httpx).

## Usage

### Start vLLM + load balancer

From this repo root:

```bash
./scripts/start_vllm_with_balancer.sh
```

- Edits: model, GPU list, and ports in the script (e.g. `MODEL_NAME`, `GPU_CONFIG`, `LB_PORT`).
- Writes `configs/vllm_endpoint_pool.txt` and `logs/vllm/vllm_pids.txt`, `logs/vllm/vllm_lb_pid.txt`.
- LB base URL: `http://localhost:<LB_PORT>` (default 7000); use `http://localhost:<LB_PORT>/v1` for OpenAI-compatible API.

### Stop vLLM and load balancer

Only stops processes whose PIDs were saved by the start script:

```bash
./scripts/stop_vllm_services.sh
```

### Load balancer only (existing backends)

If vLLM is already running on ports 8000, 8001, …:

```bash
./scripts/start_load_balancer.sh [num_instances] [base_port] [lb_port]
# Example: ./scripts/start_load_balancer.sh 4 8000 8004
```

## Config for clients

Point your LLM client at the load balancer:

- **Base URL**: `http://localhost:7000/v1` (or whatever `LB_PORT` you use).
- **Endpoint pool file** (if your app reads it): `vllm_deployment/configs/vllm_endpoint_pool.txt`.

## Health

- LB: `curl http://localhost:7000/health`
- vLLM (direct): `curl http://localhost:7001/v1/models` (per-backend port)

All paths (logs, configs, PIDs) are under this folder; no references to ReviewGrounder.
