#!/bin/bash
# deploy_vllm.sh — Deploy LLMs with vLLM and run evaluate_openrouter_models.py
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
PORT=8000
HOST="0.0.0.0"
TP_SIZE=1
MAX_MODEL_LEN=4096
GPU_UTIL=0.90

# Mapping: served-model-name (what the eval script sends) → HuggingFace model ID
declare -A MODEL_MAP=(
    ["maziyarpanahi/calme-3.2-instruct-78b"]="MaziyarPanahi/calme-3.2-instruct-78b"
    ["qwen/qwen2.5-32b-instruct"]="Qwen/Qwen2.5-32B-Instruct"
    ["qwen/qwen2.5-14b-instruct"]="Qwen/Qwen2.5-14B-Instruct"
    ["qwen/qwen2.5-7b-instruct"]="Qwen/Qwen2.5-7B-Instruct"
    ["qwen/qwen2.5-3b-instruct"]="Qwen/Qwen2.5-3B-Instruct"
    ["qwen/qwen2.5-1.5b-instruct"]="Qwen/Qwen2.5-1.5B-Instruct"
    ["qwen/qwen2.5-0.5b-instruct"]="Qwen/Qwen2.5-0.5B-Instruct"
    ["deepsseek/deepseek-r1-distill-qwen-14b"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    ["deepseek/deepseek-r1-distill-qwen-32b"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ["deepseek/deepseek-r1-distill-qwen-7b"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

VLLM_PID=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Helpers ───────────────────────────────────────────────────────────
usage() {
    cat <<'HELP'
Usage: deploy_vllm.sh [OPTIONS] [MODEL ...]

Deploy LLMs locally with vLLM (OpenAI-compatible API) and run evaluation.

Options:
  --port PORT            Server port              (default: 8000)
  --tp N                 Tensor-parallel size     (default: 1)
  --max-model-len LEN    Max context length       (default: 4096)
  --gpu-util FRAC        GPU memory utilisation   (default: 0.90)
  --all                  Deploy every model in MODEL_MAP, one at a time
  --list                 Show the model map and exit
  --deploy-only          Start server only; skip evaluation
  --eval-only URL        Run eval against an already-running server
  --models-to-run N      Only evaluate the first N models (--all mode)
  --test                 Pass --test to the eval script
  -h / --help            Show this help

Examples:
  deploy_vllm.sh --all
  deploy_vllm.sh --tp 4 qwen/qwen-2.5-72b-instruct
  deploy_vllm.sh --deploy-only meta-llama/llama-3.1-8b-instruct
  deploy_vllm.sh --eval-only http://localhost:8000
HELP
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local url="http://localhost:${PORT}/v1/models"
    local retries=0
    log "Waiting for vLLM server on port ${PORT} ..."
    while [ $retries -lt 120 ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            log "Server is ready."
            return 0
        fi
        retries=$((retries + 1))
        sleep 5
    done
    log "ERROR: server not ready after 600 s."
    return 1
}

start_vllm() {
    local hf_model="$1" served_name="$2"

    log "Launching vLLM ..."
    log "  HF model   : $hf_model"
    log "  Served as  : $served_name"
    log "  Port       : $PORT"
    log "  TP         : $TP_SIZE"

    # Try the newer `vllm serve` first, fall back to module entrypoint
    if command -v vllm >/dev/null 2>&1; then
        vllm serve "$hf_model" \
            --served-model-name "$served_name" \
            --host "$HOST" \
            --port "$PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --trust-remote-code \
            &>/tmp/vllm_server.log &
    else
        python -m vllm.entrypoints.openai.api_server \
            --model "$hf_model" \
            --served-model-name "$served_name" \
            --host "$HOST" \
            --port "$PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --trust-remote-code \
            &>/tmp/vllm_server.log &
    fi
    VLLM_PID=$!
    log "  PID: $VLLM_PID"
}

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ]; then
        log "Stopping vLLM (PID $VLLM_PID) ..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}

cleanup() { stop_vllm; }
trap cleanup EXIT INT TERM

# ── Parse args ────────────────────────────────────────────────────────
DEPLOY_ALL=false
DEPLOY_ONLY=false
EVAL_ONLY_URL=""
PASS_TEST=false
MODELS_LIMIT=""
SELECTED=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)          PORT="$2";            shift 2 ;;
        --tp)            TP_SIZE="$2";         shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2";   shift 2 ;;
        --gpu-util)      GPU_UTIL="$2";         shift 2 ;;
        --all)           DEPLOY_ALL=true;      shift   ;;
        --list)
            echo "Model map (served-name → HuggingFace ID):"
            for k in $(echo "${!MODEL_MAP[@]}" | tr ' ' '\n' | sort); do
                printf "  %-40s → %s\n" "$k" "${MODEL_MAP[$k]}"
            done
            exit 0 ;;
        --deploy-only)   DEPLOY_ONLY=true;     shift   ;;
        --eval-only)     EVAL_ONLY_URL="$2";   shift 2 ;;
        --models-to-run) MODELS_LIMIT="$2";    shift 2 ;;
        --test)          PASS_TEST=true;       shift   ;;
        -h|--help)       usage;                exit 0  ;;
        -*)              echo "Unknown flag: $1"; usage; exit 1 ;;
        *)               SELECTED+=("$1");     shift   ;;
    esac
done

# ── Eval-only mode ────────────────────────────────────────────────────
if [ -n "$EVAL_ONLY_URL" ]; then
    log "Eval-only mode — connecting to $EVAL_ONLY_URL"
    EVAL_ARGS="--base-url ${EVAL_ONLY_URL}/v1 --api-key EMPTY"
    $PASS_TEST && EVAL_ARGS="$EVAL_ARGS --test"
    python "$SCRIPT_DIR/evaluate_openrouter_models.py" $EVAL_ARGS
    exit $?
fi

# ── Build model list ──────────────────────────────────────────────────
if [ "$DEPLOY_ALL" = true ]; then
    for k in $(echo "${!MODEL_MAP[@]}" | tr ' ' '\n' | sort); do
        SELECTED+=("$k")
    done
fi

if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "ERROR: no models specified. Use --all or list model names."
    usage
    exit 1
fi

if [ -n "$MODELS_LIMIT" ]; then
    SELECTED=("${SELECTED[@]:0:$MODELS_LIMIT}")
fi

# ── Main loop ─────────────────────────────────────────────────────────
log "Will process ${#SELECTED[@]} model(s): ${SELECTED[*]}"

for served_name in "${SELECTED[@]}"; do
    hf_model="${MODEL_MAP[$served_name]:-$served_name}"

    echo ""
    log "============================================================"
    log "  $served_name"
    log "============================================================"

    start_vllm "$hf_model" "$served_name"

    if ! wait_for_server; then
        log "Server failed to start. Last 30 lines of log:"
        tail -30 /tmp/vllm_server.log
        stop_vllm
        continue
    fi

    if [ "$DEPLOY_ONLY" = false ]; then
        log "Running evaluation ..."
        EVAL_ARGS="--base-url http://localhost:${PORT}/v1 --api-key EMPTY"
        $PASS_TEST && EVAL_ARGS="$EVAL_ARGS --test"
        python "$SCRIPT_DIR/evaluate_openrouter_models.py" $EVAL_ARGS || true
    else
        log "Deploy-only mode. Server running at http://localhost:${PORT}"
        log "Press Ctrl+C to stop."
        wait "$VLLM_PID"
    fi

    stop_vllm
    log "Finished: $served_name"
done

log "All models processed."
