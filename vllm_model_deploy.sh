#!/bin/bash
# vllm-model-deploy.sh
# Deploy models from data/model_data/models_info.json using vllm
# Deploys one model at a time and supports API calls

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_INFO="${SCRIPT_DIR}/data/model_data/models_info.json"
LOG_DIR="${SCRIPT_DIR}/logs/vllm"
PID_DIR="${SCRIPT_DIR}/pids"
PORT_BASE=8000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p "$LOG_DIR" "$PID_DIR"

# Print colored message
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# List all available models
list_models() {
    if [ ! -f "$MODELS_INFO" ]; then
        log_error "Models info file not found: $MODELS_INFO"
        exit 1
    fi

    log_info "Available models from $MODELS_INFO:"
    echo ""
    python3 -c "
import json
with open('$MODELS_INFO', 'r') as f:
    models = json.load(f)
for i, (name, info) in enumerate(models.items(), 1):
    print(f\"  {i}. {name}\")
    print(f\"     - Base model: {info.get('base_model', 'N/A')}\")
    print(f\"     - BBH Acc: {info.get('bbh_acc', 'N/A'):.4f}\")
    print(f\"     - CO2 Cost: {info.get('co2_cost', 'N/A')}\")
    print()
"
}

# Get model name by index
get_model_name_by_index() {
    local index=$1
    python3 -c "
import json
with open('$MODELS_INFO', 'r') as f:
    models = json.load(f)
names = list(models.keys())
if 1 <= $index <= len(names):
    print(names[$index - 1])
else:
    print('')
"
}

# Get model name directly
get_model_name() {
    python3 -c "
import json
with open('$MODELS_INFO', 'r') as f:
    models = json.load(f)
if '$1' in models:
    print('$1')
else:
    # Try partial match
    matches = [k for k in models.keys() if '$1' in k]
    if matches:
        print(matches[0])
    else:
        print('')
"
}

# Check if a model is running
is_running() {
    local model_name=$1
    local pid_file="${PID_DIR}/${model_name//\//_}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        else
            rm -f "$pid_file"
        fi
    fi
    return 1
}

# Get port for a model
get_port() {
    local model_name=$1
    python3 -c "
import json
with open('$MODELS_INFO', 'r') as f:
    models = json.load(f)
names = list(models.keys())
if '$model_name' in names:
    idx = names.index('$model_name')
    print($PORT_BASE + idx)
elif '$model_name' in [n for n in names]:
    idx = [n for n in names].index('$model_name')
    print($PORT_BASE + idx)
else:
    print(-1)
"
}

# Start a model
start_model() {
    local model_name=$1
    local port=${2:-}

    if [ -z "$model_name" ]; then
        log_error "Model name required"
        echo "Usage: $0 start <model_name_or_index>"
        exit 1
    fi

    # Handle index
    if [[ "$model_name" =~ ^[0-9]+$ ]]; then
        model_name=$(get_model_name_by_index "$model_name")
        if [ -z "$model_name" ]; then
            log_error "Invalid model index"
            list_models
            exit 1
        fi
    else
        # Try to resolve partial name
        local resolved
        resolved=$(python3 -c "
import json
with open('$MODELS_INFO', 'r') as f:
    models = json.load(f)
matches = [k for k in models.keys() if '$model_name' in k]
if len(matches) == 1:
    print(matches[0])
elif len(matches) > 1:
    print('AMBIGUOUS:' + ','.join(matches))
else:
    print('')
")
        if [[ "$resolved" == "AMBIGUOUS:"* ]]; then
            log_error "Ambiguous model name. Matches:"
            echo "${resolved#AMBIGUOUS:}" | tr ',' '\n' | sed 's/^/  /'
            exit 1
        elif [ -n "$resolved" ]; then
            model_name="$resolved"
        fi
    fi

    # Check if already running
    if is_running "$model_name"; then
        local existing_port
        existing_port=$(get_port "$model_name")
        log_warn "Model '$model_name' is already running on port $existing_port"
        return 0
    fi

    # Get port for this model
    if [ -z "$port" ]; then
        port=$(python3 -c "
import json
with open('$MODELS_INFO', 'r') as f:
    models = json.load(f)
names = list(models.keys())
if '$model_name' in names:
    print($PORT_BASE + names.index('$model_name'))
else:
    print(-1)
")
        if [ "$port" -eq -1 ]; then
            log_error "Model '$model_name' not found in $MODELS_INFO"
            exit 1
        fi
    fi

    # Check if port is in use
    if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        log_error "Port $port is already in use"
        log_info "Stop the existing service first"
        exit 1
    fi

    log_info "Starting model: $model_name on port $port"

    # Create log file
    local log_file="${LOG_DIR}/${model_name//\//_}.log"
    local pid_file="${PID_DIR}/${model_name//\//_}.pid"

    # Start vllm server in background
    nohup vllm serve "$model_name" \
        --port "$port" \
        --host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 4096 \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    log_info "Model starting with PID $pid"
    log_info "Logs: $log_file"

    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    local max_attempts=60
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            log_info "Model '$model_name' is ready at http://localhost:$port"
            echo ""
            echo "API Examples:"
            echo "  curl -X POST http://localhost:$port/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
        if [ $((attempt % 10)) -eq 0 ]; then
            log_info "Still waiting... (${attempt}/${max_attempts})"
        fi
    done

    log_error "Model failed to start within timeout. Check logs: $log_file"
    return 1
}

# Stop a model
stop_model() {
    local model_name=$1

    if [ -z "$model_name" ]; then
        log_error "Model name required"
        echo "Usage: $0 stop <model_name_or_index>"
        exit 1
    fi

    # Handle index
    if [[ "$model_name" =~ ^[0-9]+$ ]]; then
        model_name=$(get_model_name_by_index "$model_name")
    fi

    local pid_file="${PID_DIR}/${model_name//\//_}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping model '$model_name' (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
            log_info "Model '$model_name' stopped"
        else
            rm -f "$pid_file"
            log_warn "Model '$model_name' was not running"
        fi
    else
        log_error "No PID file found for '$model_name'"
        log_info "Run '$0 status' to see running models"
        exit 1
    fi
}

# Stop all models
stop_all() {
    log_info "Stopping all running models..."
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local model_name=$(basename "$pid_file" .pid)
            if kill -0 "$pid" 2>/dev/null; then
                log_info "Stopping $model_name (PID $pid)..."
                kill "$pid" 2>/dev/null || true
            fi
        fi
    done
    rm -f "$PID_DIR"/*.pid
    log_info "All models stopped"
}

# Show status
show_status() {
    echo ""
    log_info "Running Models:"
    echo ""

    local found=0
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local model_name=$(basename "$pid_file" .pid | tr '_' '/')
            if kill -0 "$pid" 2>/dev/null; then
                local port=$(get_port "$model_name")
                local health="unknown"
                if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
                    health="healthy"
                else
                    health="starting/unhealthy"
                fi
                echo -e "  ${GREEN}●${NC} $model_name"
                echo "    PID: $pid"
                echo "    Port: $port"
                echo "    Health: $health"
                found=1
            fi
        fi
    done

    if [ $found -eq 0 ]; then
        echo "  No models currently running"
    fi

    echo ""
}

# Call a model via API
call_model() {
    local model_name=$1
    shift
    local prompt="$*"

    if [ -z "$model_name" ] || [ -z "$prompt" ]; then
        log_error "Usage: $0 call <model_name_or_index> <prompt>"
        exit 1
    fi

    # Handle index
    if [[ "$model_name" =~ ^[0-9]+$ ]]; then
        model_name=$(get_model_name_by_index "$model_name")
    fi

    if ! is_running "$model_name"; then
        log_error "Model '$model_name' is not running"
        log_info "Start it with: $0 start $model_name"
        exit 1
    fi

    local port=$(get_port "$model_name")

    curl -s -X POST "http://localhost:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model_name\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}]}"
}

# Stream call
stream_call() {
    local model_name=$1
    shift
    local prompt="$*"

    if [ -z "$model_name" ] || [ -z "$prompt" ]; then
        log_error "Usage: $0 stream <model_name_or_index> <prompt>"
        exit 1
    fi

    if ! is_running "$model_name"; then
        log_error "Model '$model_name' is not running"
        exit 1
    fi

    local port=$(get_port "$model_name")

    curl -s -N "http://localhost:$port/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model_name\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}], \"stream\": true}"
}

# Show logs
show_logs() {
    local model_name=$1

    if [ -z "$model_name" ]; then
        log_error "Usage: $0 logs <model_name_or_index>"
        exit 1
    fi

    if [[ "$model_name" =~ ^[0-9]+$ ]]; then
        model_name=$(get_model_name_by_index "$model_name")
    fi

    local log_file="${LOG_DIR}/${model_name//\//_}.log"

    if [ -f "$log_file" ]; then
        tail -100 "$log_file"
    else
        log_error "No log file found for '$model_name'"
        exit 1
    fi
}

# Interactive mode
interactive() {
    echo ""
    log_info "vllm Model Deployer - Interactive Mode"
    echo "========================================"
    echo ""

    while true; do
        echo -n "vllm> "
        read -r cmd arg1 arg2

        case "$cmd" in
            start)
                start_model "$arg1"
                ;;
            stop)
                stop_model "$arg1"
                ;;
            list)
                list_models
                ;;
            status)
                show_status
                ;;
            call)
                echo -n "Enter prompt: "
                read -r prompt
                call_model "$arg1" "$prompt"
                ;;
            stream)
                echo -n "Enter prompt: "
                read -r prompt
                stream_call "$arg1" "$prompt"
                ;;
            logs)
                show_logs "$arg1"
                ;;
            help)
                echo "Commands: start, stop, list, status, call, stream, logs, help, exit"
                ;;
            exit|quit)
                break
                ;;
            "")
                ;;
            *)
                log_error "Unknown command: $cmd"
                echo "Commands: start, stop, list, status, call, stream, logs, help, exit"
                ;;
        esac
    done
}

# Print usage
usage() {
    echo "vllm Model Deployer"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  list              List all available models"
    echo "  start <model>     Start a model (by name or index)"
    echo "  stop <model>      Stop a running model"
    echo "  stop-all          Stop all running models"
    echo "  status            Show running models"
    echo "  call <model> <prompt>  Call model via API"
    echo "  stream <model> <prompt>  Stream call to model"
    echo "  logs <model>      Show logs for a model"
    echo "  interactive       Enter interactive mode"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 start 1"
    echo "  $0 start MaziyarPanahi/calme-3.2-instruct-78b"
    echo "  $0 call 1 \"What is 2+2?\""
    echo "  $0 logs MaziyarPanahi/calme-3.2-instruct-78b"
    echo ""
}

# Main command dispatcher
case "${1:-}" in
    list)
        list_models
        ;;
    start)
        start_model "$2" "$3"
        ;;
    stop)
        stop_model "$2"
        ;;
    stop-all)
        stop_all
        ;;
    status)
        show_status
        ;;
    call)
        call_model "$2" "$3"
        ;;
    stream)
        stream_call "$2" "$3"
        ;;
    logs)
        show_logs "$2"
        ;;
    interactive|i)
        interactive
        ;;
    help|--help|-h)
        usage
        ;;
    "")
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac