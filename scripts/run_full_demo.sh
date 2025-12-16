#!/bin/bash
#
# PlanetServe Full Demo with BFT Consensus
#
# Architecture:
#   Model Node (llama.cpp) ←─S-IDA─→ Verifier Node (llama.cpp) ──HTTP──→ Java Tendermint (BFT)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
LOG_DIR="$PROJECT_DIR/logs/demo"
PID_DIR="$PROJECT_DIR/.pids"
CONFIG_FILE="$PROJECT_DIR/configs/demo_local.yaml"

# Paths
MODEL_PATH="$PROJECT_DIR/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
TENDERMINT_DIR="$PROJECT_DIR/planetllm_tendermint"

# Ports
LLAMA_SERVER_PORT=8080
TENDERMINT_PORT=26657
MODEL_NODE_PORT=9000
VERIFIER_PORT=9100
USER_PORT=9200

# Relay ports for 4 paths × 3 hops
RELAY_PORTS=(9300 9301 9302 9310 9311 9312 9320 9321 9322 9330 9331 9332)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

ensure_dirs() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$PID_DIR"
}

check_build() {
    if [ ! -d "$BUILD_DIR" ]; then
        log_error "Build directory not found. Run: $0 build"
        exit 1
    fi
    
    for exe in demo_model_node demo_user_node demo_verifier_node relay_node; do
        if [ ! -f "$BUILD_DIR/$exe" ]; then
            log_error "Executable $exe not found. Run: $0 build"
            exit 1
        fi
    done
}

check_model() {
    if [ ! -f "$MODEL_PATH" ]; then
        log_error "Model file not found: $MODEL_PATH"
        log_info "Download with:"
        echo "  mkdir -p $PROJECT_DIR/models && cd $PROJECT_DIR/models"
        echo "  wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        exit 1
    fi
}

  
# Process Management
  

start_process() {
    local name=$1
    local cmd=$2
    local log_file=$3
    
    nohup $cmd > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$PID_DIR/${name}.pid"
    sleep 0.3
    
    if kill -0 $pid 2>/dev/null; then
        log_info "Started $name (PID: $pid)"
    else
        log_error "$name failed to start. Check $log_file"
        return 1
    fi
}

stop_process() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
            sleep 0.2
            kill -0 $pid 2>/dev/null && kill -9 $pid 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

check_process() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} $name (PID: $pid)"
            return 0
        fi
    fi
    echo -e "  ${RED}○${NC} $name"
    return 1
}

  
# Service Start/Stop
  

start_llama_server() {
    log_section "Starting llama-server (for Java Tendermint)"
    
    if curl -s "http://localhost:$LLAMA_SERVER_PORT/health" 2>/dev/null | grep -q "ok"; then
        log_info "llama-server already running on port $LLAMA_SERVER_PORT"
        return 0
    fi
    
    check_model
    
    log_info "Starting llama-server on port $LLAMA_SERVER_PORT..."
    nohup llama-server \
        -m "$MODEL_PATH" \
        --port "$LLAMA_SERVER_PORT" \
        --ctx-size 2048 \
        > "$LOG_DIR/llama_server.log" 2>&1 &
    
    echo $! > "$PID_DIR/llama_server.pid"
    
    log_info "Waiting for llama-server..."
    for i in {1..60}; do
        if curl -s "http://localhost:$LLAMA_SERVER_PORT/health" 2>/dev/null | grep -q "ok"; then
            log_info "llama-server is ready!"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    echo ""
    
    log_error "llama-server failed to start"
    return 1
}

start_java_tendermint() {
    log_section "Starting Java Tendermint (BFT Consensus)"
    
    if [ ! -d "$TENDERMINT_DIR" ]; then
        log_error "Tendermint directory not found: $TENDERMINT_DIR"
        return 1
    fi
    
    cd "$TENDERMINT_DIR"
    
    log_info "Building Java Tendermint..."
    ./gradlew build -q 2>&1 || {
        log_error "Failed to build Java Tendermint"
        return 1
    }
    
    log_info "Starting Java Tendermint app..."
    nohup ./gradlew run -q > "$LOG_DIR/tendermint_app.log" 2>&1 &
    local gradle_pid=$!
    echo $gradle_pid > "$PID_DIR/tendermint_app.pid"
    
    # Wait for app to be ready
    log_info "Waiting for Tendermint to be ready..."
    sleep 5
    
    log_info "Java Tendermint started (Gradle PID: $gradle_pid)"
    cd "$PROJECT_DIR"
}

start_relay_nodes() {
    log_info "Starting 12 relay nodes..."
    for port in "${RELAY_PORTS[@]}"; do
        start_process "relay_$port" \
            "$BUILD_DIR/relay_node --port $port --config $CONFIG_FILE" \
            "$LOG_DIR/relay_$port.log"
    done
}

start_model_node() {
    log_info "Starting Model Node (with local llama.cpp LLM)..."
    start_process "model_node" \
        "$BUILD_DIR/demo_model_node --config $CONFIG_FILE" \
        "$LOG_DIR/model_node.log"
}

start_all() {
    ensure_dirs
    check_build
    check_model
    
    log_section "Starting PlanetServe Full Demo"
    
    # 1. Start llama-server (for Java Tendermint's perplexity)
    start_llama_server
    
    # 2. Start Java Tendermint (BFT consensus)
    start_java_tendermint
    
    # 3. Start relay network
    log_section "Starting C++ Relay Network"
    start_relay_nodes
    
    # 4. Start model node (with its own llama.cpp)
    log_section "Starting C++ Model Node"
    start_model_node
    
    log_section "Full Demo Started"
    echo "Services:"
    echo "  - llama-server:     http://localhost:$LLAMA_SERVER_PORT (for Tendermint)"
    echo "  - Java Tendermint:  BFT consensus layer"
    echo "  - Model Node:       port $MODEL_NODE_PORT (with local llama.cpp)"
    echo "  - Relay Network:    ports 9300-9332 (12 nodes)"
    echo ""
    echo "Architecture:"
    echo "  Model Node (llama.cpp) ←─S-IDA─→ Verifier (llama.cpp) ──HTTP──→ Tendermint (BFT)"
    echo ""
    echo "Commands:"
    echo "  $0 test 5          # Run 5 challenges with BFT consensus"
    echo "  $0 user --prompt   # Send user prompt via S-IDA"
    echo "  $0 status          # Check all services"
    echo "  $0 stop            # Stop everything"
}

stop_all() {
    log_section "Stopping All Services"
    
    # Stop C++ components
    stop_process "model_node"
    for port in "${RELAY_PORTS[@]}"; do
        stop_process "relay_$port"
    done
    
    # Stop Java Tendermint
    if [ -f "$PID_DIR/tendermint_app.pid" ]; then
        local pid=$(cat "$PID_DIR/tendermint_app.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping Java Tendermint..."
            pkill -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_DIR/tendermint_app.pid"
    fi
    
    # Stop llama-server
    if [ -f "$PID_DIR/llama_server.pid" ]; then
        local pid=$(cat "$PID_DIR/llama_server.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping llama-server..."
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_DIR/llama_server.pid"
    fi
    
    # Cleanup
    pkill -f "demo_model_node" 2>/dev/null || true
    pkill -f "demo_verifier_node" 2>/dev/null || true
    pkill -f "relay_node" 2>/dev/null || true
    pkill -f "llama-server.*$LLAMA_SERVER_PORT" 2>/dev/null || true
    pkill -f "gradlew.*run" 2>/dev/null || true
    
    log_info "All services stopped"
}

show_status() {
    log_section "Service Status"
    
    echo "llama-server (port $LLAMA_SERVER_PORT):"
    if curl -s "http://localhost:$LLAMA_SERVER_PORT/health" 2>/dev/null | grep -q "ok"; then
        echo -e "  ${GREEN}●${NC} Running"
    else
        echo -e "  ${RED}○${NC} Not running"
    fi
    
    echo ""
    echo "Java Tendermint:"
    check_process "tendermint_app" || true
    
    echo ""
    echo "Model Node:"
    check_process "model_node" || true
    
    echo ""
    echo "Relay Nodes:"
    for port in "${RELAY_PORTS[@]}"; do
        check_process "relay_$port" || true
    done
}

  
# User & Verifier Commands
  

run_user() {
    check_build
    log_section "Running User Node"
    "$BUILD_DIR/demo_user_node" --config "$CONFIG_FILE" "$@"
}

run_verifier() {
    check_build
    log_section "Running Verifier Node"
    "$BUILD_DIR/demo_verifier_node" --config "$CONFIG_FILE" "$@"
}

  
# E2E Test with BFT Consensus
  

run_e2e_test() {
    local NUM="${1:-5}"
    
    log_section "E2E Test with BFT Consensus ($NUM challenges)"

    # TODO: change to random generate challenge 
    CHALLENGES=(
        "What is the capital of France?"
        "What is 2 + 2?"
        "Complete: The quick brown fox..."
        "What color is the sky?"
        "What is a neural network?"
        "Largest planet in solar system?"
        "Who wrote Romeo and Juliet?"
        "Chemical symbol for water?"
        "How many continents on Earth?"
        "Speed of light?"
    )
    
    TOTAL_SCORE=0
    PASSED=0
    
    # Start new epoch in Tendermint
    log_info "Starting new epoch in Tendermint..."
    curl -s -X POST "http://localhost:$LLAMA_SERVER_PORT/../tx" \
        -H "Content-Type: application/json" \
        -d '{"type":"start_epoch"}' 2>/dev/null || true
    
    for i in $(seq 1 $NUM); do
        IDX=$((i - 1))
        CHALLENGE="${CHALLENGES[$IDX]}"
        
        echo ""
        echo "╔═══════════════════════════════════════════════════════════╗"
        echo "║  Challenge $i / $NUM                                          ║"
        echo "╚═══════════════════════════════════════════════════════════╝"
        echo ""
        echo "Challenge: \"$CHALLENGE\""
        echo ""
        
        # Run verifier with Tendermint submission
        # The verifier will:
        # 1. Send challenge via S-IDA to model node
        # 2. Receive response
        # 3. Calculate perplexity with local LLM
        # 4. Submit result to Tendermint for BFT consensus
        OUTPUT=$("$BUILD_DIR/demo_verifier_node" \
            --config "$CONFIG_FILE" \
            --tendermint "localhost:$LLAMA_SERVER_PORT" \
            --challenge "$CHALLENGE" 2>&1) || true
        
        # Parse output
        RESPONSE=$(echo "$OUTPUT" | grep -A1 "=== Response ===" | tail -1 | head -c 60)
        PPL=$(echo "$OUTPUT" | grep "Perplexity:" | head -1 | awk '{print $2}')
        CRED=$(echo "$OUTPUT" | grep "Credibility" | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        BFT_STATUS=$(echo "$OUTPUT" | grep -i "tendermint\|bft\|submitted" | head -1)
        
        echo "Response: ${RESPONSE}..."
        echo "Perplexity: ${PPL:-N/A}"
        echo "Credibility: ${CRED:-N/A}"
        [ -n "$BFT_STATUS" ] && echo "BFT: $BFT_STATUS"
        
        if [ -n "$CRED" ]; then
            TOTAL_SCORE=$(echo "$TOTAL_SCORE + $CRED" | bc -l 2>/dev/null || echo "$TOTAL_SCORE")
            PASSED=$((PASSED + 1))
        fi
        
        sleep 1
    done
    
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "                    TEST SUMMARY                            "
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "Challenges: $PASSED / $NUM completed"
    if [ "$PASSED" -gt 0 ]; then
        AVG=$(echo "scale=4; $TOTAL_SCORE / $PASSED" | bc -l 2>/dev/null || echo "N/A")
        echo "Average Credibility: $AVG"
    fi
    echo ""
    
    # Query Tendermint for final reputations
    log_info "Querying Tendermint for reputation status..."
    curl -s "http://localhost:$LLAMA_SERVER_PORT/../query?path=reputations" 2>/dev/null || echo "  (Tendermint query not available)"
    
    echo ""
    echo "✓ E2E test with BFT consensus completed"
}

run_quick_test() {
    local CHALLENGE="${1:-What is 2+2?}"
    log_section "Quick Test (with BFT)"
    log_info "Challenge: $CHALLENGE"
    echo ""
    "$BUILD_DIR/demo_verifier_node" \
        --config "$CONFIG_FILE" \
        --tendermint "localhost:$LLAMA_SERVER_PORT" \
        --challenge "$CHALLENGE"
}

  
# Build
  

build_demo() {
    log_section "Building Demo"
    
    # Build C++
    log_info "Building C++ components..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake ..
    make demo_all -j$(nproc)
    cd "$PROJECT_DIR"
    
    # Build Java
    log_info "Building Java Tendermint..."
    cd "$TENDERMINT_DIR"
    ./gradlew build -q
    cd "$PROJECT_DIR"
    
    log_info "Build complete!"
}

  
# Usage
  

print_usage() {
    cat << 'EOF'
PlanetServe Full Demo with BFT Consensus

Usage:
  ./run_full_demo.sh start              Start all services
  ./run_full_demo.sh stop               Stop all services
  ./run_full_demo.sh status             Check service status
  ./run_full_demo.sh test [N]           Run N challenges with BFT (default: 5)
  ./run_full_demo.sh quick [prompt]     Quick single test with BFT
  ./run_full_demo.sh user [args]        Run user node (S-IDA)
  ./run_full_demo.sh verifier [args]    Run verifier node
  ./run_full_demo.sh build              Build C++ and Java
  ./run_full_demo.sh logs               Show recent logs

EOF
}

  
# Main
  

ensure_dirs

case "${1:-}" in
    start)      start_all ;;
    stop)       stop_all ;;
    status)     show_status ;;
    test)       run_e2e_test "${2:-5}" ;;
    quick)      run_quick_test "${2:-What is 2+2?}" ;;
    user)       shift; run_user "$@" ;;
    verifier)   shift; run_verifier "$@" ;;
    build)      build_demo ;;
    logs)
        log_section "Recent Logs"
        echo "=== llama-server ===" 
        tail -15 "$LOG_DIR/llama_server.log" 2>/dev/null || echo "  No logs"
        echo ""
        echo "=== Java Tendermint ==="
        tail -15 "$LOG_DIR/tendermint_app.log" 2>/dev/null || echo "  No logs"
        echo ""
        echo "=== Model Node ==="
        tail -15 "$LOG_DIR/model_node.log" 2>/dev/null || echo "  No logs"
        ;;
    *)          print_usage ;;
esac
