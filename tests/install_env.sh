#!/bin/bash
#
# Install vllm_gcu Python environment: dependencies and wheel package.
# Requires: python3, sudo (for system-wide pip install).
#
# Usage:
#   bash install_env.sh           # Quiet pip output (default)
#   bash install_env.sh --debug   # Show full pip install logs
#
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_step()  { echo -e "${YELLOW}[STEP]${NC} $*"; }
log_ok()    { echo -e "${GREEN}${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

DEBUG=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--debug) DEBUG=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--debug|-d]"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            echo "Usage: $0 [--debug|-d]"
            exit 1
            ;;
    esac
done

run_pip() {
    if [ "$DEBUG" -eq 1 ]; then
        sudo python3 -m pip "$@"
    else
        local log_file
        log_file="$(mktemp /tmp/vllm_gcu_pip_XXXXXX.log)"
        if sudo python3 -m pip "$@" > "$log_file" 2>&1; then
            rm -f "$log_file"
        else
            echo -e "${RED}[ERROR]${NC} pip install failed:"
            cat "$log_file"
            log_info "Re-run with --debug to stream pip output live."
            rm -f "$log_file"
            exit 1
        fi
    fi
}

TOTAL_STEPS=2
STEP=0

next_step() {
    STEP=$((STEP + 1))
    echo ""
    log_step "[$STEP/$TOTAL_STEPS] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/.. && pwd)"

echo ""
echo "============================================"
echo "  vllm_gcu Environment Installer"
echo "============================================"
log_info "Working directory: $REPO_ROOT"
log_info "Python:          $(python3 --version 2>&1 || echo 'not found')"
if [ "$DEBUG" -eq 1 ]; then
    log_info "Debug mode:      enabled (pip output visible)"
fi
echo ""

cd "$REPO_ROOT"

REQUIREMENTS="$REPO_ROOT/requirements.txt"
if [ ! -f "$REQUIREMENTS" ]; then
    echo -e "${RED}[ERROR]${NC} requirements.txt not found: $REQUIREMENTS"
    exit 1
fi

WHEELS=(dist/vllm_gcu*.whl)
if [ ! -f "${WHEELS[0]}" ]; then
    echo -e "${RED}[ERROR]${NC} No vllm_gcu wheel found in $REPO_ROOT/dist/"
    log_info "Build the wheel first, e.g.: bash .pipeline/build.sh"
    exit 1
fi

next_step "Install Python dependencies"
log_info "Installing from: $REQUIREMENTS"
log_warn "You may be prompted for your sudo password."
run_pip install -r "$REQUIREMENTS"
log_ok "Dependencies installed"

next_step "Install vllm_gcu wheel"
log_info "Wheel package(s):"
ls -lh "${WHEELS[@]}"
echo ""
log_warn "You may be prompted for your sudo password again."
cd /home/pypi_packages
run_pip install vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl
cd -
run_pip install "${WHEELS[@]}"
log_ok "vllm_gcu installed: $(python3 -c 'import vllm_gcu; print(vllm_gcu.__file__)' 2>/dev/null || echo '(import check skipped)')"

echo ""
echo "============================================"
echo -e "  ${GREEN}Environment installation finished.${NC}"
echo "============================================"
echo ""
