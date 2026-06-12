#!/bin/bash
#
# Install Enflame driver from the vllm_gcu Docker image.
# Requires: docker, sudo (for driver .run installer).
#
# Usage:
#   bash install_driver.sh           # Quiet driver installer output (default)
#   bash install_driver.sh --debug   # Show full driver installer logs
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

run_installer() {
    local installer="$1"
    if [ "$DEBUG" -eq 1 ]; then
        sudo "./${installer}" --no-auto-load --no-dkms
    else
        local log_file
        log_file="$(mktemp /tmp/torch_gcu_driver_XXXXXX.log)"
        if sudo "./${installer}" --no-auto-load --no-dkms > "$log_file" 2>&1; then
            rm -f "$log_file"
        else
            echo -e "${RED}[ERROR]${NC} Driver installation failed:"
            cat "$log_file"
            log_info "Re-run with --debug to stream installer output live."
            rm -f "$log_file"
            exit 1
        fi
    fi
}

DOCKER_IMAGE="registry-egc.enflame-tech.com/artifacts/vllm_gcu:v0.11.0-TR3.7.107-ubuntu2204"
CONTAINER_NAME="tmp_container"
TOTAL_STEPS=4
STEP=0

next_step() {
    STEP=$((STEP + 1))
    echo ""
    log_step "[$STEP/$TOTAL_STEPS] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/../.. && pwd)"

echo ""
echo "============================================"
echo "  Enflame Driver Installer"
echo "============================================"
log_info "Working directory: $REPO_ROOT"
log_info "Docker image:      $DOCKER_IMAGE"
if [ "$DEBUG" -eq 1 ]; then
    log_info "Debug mode:      enabled (installer output visible)"
fi
echo ""

cd "$REPO_ROOT"

# Remove leftover container from a previous failed run
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log_warn "Removing existing container '$CONTAINER_NAME' from a previous run..."
    docker rm -f "$CONTAINER_NAME" > /dev/null
    log_ok "Old container removed"
fi

next_step "Pull driver files from Docker image"
log_info "Creating temporary container (this may take a while on first run)..."
docker run --name "$CONTAINER_NAME" "$DOCKER_IMAGE"
log_ok "Container created: $CONTAINER_NAME"

next_step "Copy driver package to host"
mkdir -p tmp
log_info "Copying /enflame from container to ./tmp/ ..."
docker cp "$CONTAINER_NAME:/enflame" tmp/
log_ok "Driver files copied to $REPO_ROOT/tmp/enflame"

next_step "Install driver (requires sudo)"
DRIVER_DIR="$REPO_ROOT/tmp/enflame/driver"
cd "$DRIVER_DIR"
log_info "Driver directory: $DRIVER_DIR"
log_info "Available installer(s):"
ls -lh enflame-x86_64*.run 2>/dev/null || ls -lh
echo ""
log_warn "You may be prompted for your sudo password."
INSTALLER=(enflame-x86_64*.run)
if [ ! -f "${INSTALLER[0]}" ]; then
    echo -e "${RED}[ERROR]${NC} No enflame-x86_64*.run installer found in $DRIVER_DIR"
    exit 1
fi
log_info "Running: sudo ./${INSTALLER[0]} --no-auto-load --no-dkms"
run_installer "${INSTALLER[0]}"
log_ok "Driver installed successfully"

next_step "Clean up temporary files"
cd "$REPO_ROOT"
log_info "Removing ./tmp and container $CONTAINER_NAME ..."
rm -rf tmp
docker rm "$CONTAINER_NAME" > /dev/null
log_ok "Cleanup complete"

echo ""
echo "============================================"
echo -e "  ${GREEN}Driver installation finished.${NC}"
echo "============================================"
echo ""
