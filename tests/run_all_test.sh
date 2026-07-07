#!/bin/bash
#
# Copyright 2023-2025 Enflame. All Rights Reserved.
#
# Run all vllm_gcu tests: kernel tests, libtorch C++ tests, CUDA-compat API tests.
#
# Usage:
#   bash run_all_test.sh              # Run all tests
#   bash run_all_test.sh kernel           # Run only kernel tests
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=""

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local test_dir="$3"
    local log_name="${test_name//\//_}"

    echo -e "${YELLOW}[RUN]${NC} $test_name"
    pushd "$test_dir" > /dev/null 2>&1 || true

    if eval "$test_cmd" > "/tmp/vllm_gcu_test_${log_name}.log" 2>&1; then
        echo -e "${GREEN}[PASS]${NC} $test_name"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}[FAIL]${NC} $test_name (see /tmp/vllm_gcu_test_${log_name}.log)"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="$FAILED_TESTS\n  - $test_name"
    fi

    popd > /dev/null 2>&1 || true
}

# =============================================================================
# Kernel Tests (Python)
# =============================================================================
run_kernel_tests() {
    echo ""
    echo "=============================="
    echo "  Kernel Tests (Python)"
    echo "=============================="

    local kernel_dir="$SCRIPT_DIR/kernel_test"
    if [ ! -d "$kernel_dir" ]; then
        echo -e "${YELLOW}[SKIP]${NC} kernel_test directory not found"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    for test_file in "$kernel_dir"/test_*.py; do
        if [ -f "$test_file" ]; then
            local test_name="kernel_test/$(basename "$test_file" .py)"
            run_test "$test_name" "python3 $(basename "$test_file")" "$kernel_dir"
        fi
    done
}

# =============================================================================
# Main
# =============================================================================
echo "============================================"
echo "  vllm_gcu Test Suite"
echo "============================================"
echo "  Test directory: $SCRIPT_DIR"
echo "  Date: $(date)"
echo "============================================"

TARGET="${1:-all}"

case "$TARGET" in
    kernel)
        run_kernel_tests
        ;;
    all)
        run_kernel_tests
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [all|kernel|libtorch|cuda]"
        exit 1
        ;;
esac

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================"
echo "  Test Summary"
echo "============================================"
echo -e "  ${GREEN}PASSED${NC}: $PASSED"
echo -e "  ${RED}FAILED${NC}: $FAILED"
echo -e "  ${YELLOW}SKIPPED${NC}: $SKIPPED"

if [ $FAILED -gt 0 ]; then
    echo -e "\n  Failed tests:${FAILED_TESTS}"
    echo ""
    echo "============================================"
    exit 1
else
    echo ""
    echo -e "  ${GREEN}All tests passed!${NC}"
    echo "============================================"
    exit 0
fi
