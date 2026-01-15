#!/usr/bin/env python3
"""
Simple pytest for vLLM deepseek-r1 6-layer benchmark
Usage: python -m pytest -sv test_deepseek_r1_6layers.py
"""

import os
import sys
import time
import subprocess
import signal
import pytest
import re
from pathlib import Path


# Configuration - modify these as needed
MODEL_PATH = "/home/jenkins/inference/libra/vllm/deepseek-v3.2/"
SERVER_LOG = "server_v1_deepseek_r1_6layers_mtp_fusion_log.txt"
CLIENT_LOG = "client_v1_deepseek_r1_6layers_mtp_fusion_log.txt"
SERVER_TIMEOUT = 500
NUM_PROMPTS = 32
MAX_CONCURRENCY = 8

# Performance requirements
MIN_OUTPUT_TOKEN_THROUGHPUT = 200  # tok/s  # ci使用mc残血卡151

# Global variable to track server process
server_process = None


def setup_environment():
    """Setup environment variables"""
    env_vars = {
        "PYTORCH_EFML_BASED_GCU_CHECK": "1",
        "TORCHGCU_INDUCTOR_ENABLE": "0",
        "TORCH_ECCL_AVOID_RECORD_STREAMS": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_USE_V1": "1",
        "VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION": "1",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✓ {key}={value}")


def print_server_log():
    """Print server log content for debugging"""
    try:
        if Path(SERVER_LOG).exists():
            print(f"\n📋 =============== {SERVER_LOG} Content ===============")
            with open(SERVER_LOG, 'r') as f:
                log_content = f.read()
                if log_content.strip():
                    print(log_content)
                else:
                    print("(Log file is empty)")
            print("=" * (30 + len(SERVER_LOG)))
        else:
            print(f"\n⚠️ Server log file not found: {SERVER_LOG}")
    except Exception as e:
        print(f"\n❌ Error reading server log: {e}")


def cleanup_server():
    """Cleanup server process"""
    global server_process
    if server_process:
        print("🛑 Stopping server...")
        try:
            server_process.terminate()
            server_process.wait(timeout=10)
            print("✅ Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("⚠️ Forcing server termination...")
            server_process.kill()
            server_process.wait()
            print("✅ Server forcefully terminated")
        except Exception as e:
            print(f"❌ Error stopping server: {e}")
        finally:
            server_process = None


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print(f"\n🛑 Received signal {signum}, cleaning up...")
    cleanup_server()
    sys.exit(0)


# Setup signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    """Session setup and teardown"""
    print("\n🔧 Setting up test session...")
    setup_environment()
    yield
    print("\n🧹 Cleaning up test session...")
    cleanup_server()


@pytest.fixture(autouse=True)
def setup_test():
    """Test setup and teardown"""
    yield
    # Cleanup after each test
    cleanup_server()


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing 'datasets' package...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "datasets", "--break-system-packages"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Successfully installed 'datasets' package")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install 'datasets' package: {e}")
        pytest.fail("Failed to install datasets package")


def check_model_path_exists():
    """Verify model path exists"""
    print(f"\n📁 Checking model path: {MODEL_PATH}")
    
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model path does not exist: {MODEL_PATH}")
    
    print("✅ Model path exists")


def start_server():
    """Start vLLM server"""
    global server_process
    
    print(f"\n🚀 Starting server with model: {MODEL_PATH}")
    
    server_cmd = [
        # "vllm", "serve",
        "coverage", "run", "--parallel-mode", "-m", "vllm.entrypoints.cli.main", "serve",
        f"--model={MODEL_PATH}",
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--max-model-len", "2048",
        "--dtype", "bfloat16",
        "--quantization", "fp8",
        "--seed", "0",
        "--gpu-memory-utilization", "0.9",
        "--compilation-config", '{"cudagraph_mode": "FULL"}',
        "--hf-overrides", '{"num_hidden_layers":6}',
        "--cuda-graph-sizes", "8",
        "--max-num-seqs", "256",
        "--speculative-config", '{"method":"deepseek_mtp", "num_speculative_tokens":1}'
    ]
    
    print(f"🔧 Server command: {' '.join(server_cmd)}")
    
    try:
        with open(SERVER_LOG, 'w') as log_file:
            server_process = subprocess.Popen(
                server_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        print(f"✅ Server started with PID: {server_process.pid}")
        print(f"📝 Server logs: {SERVER_LOG}")
        
    except Exception as e:
        pytest.fail(f"Failed to start server: {e}")


def wait_for_server_ready():
    """Wait for server to be ready by checking log file"""
    global server_process
    
    print(f"⏳ Waiting for server to be ready (max {SERVER_TIMEOUT} seconds)...")
    startup_complete_message = "INFO:     Application startup complete."
    
    start_time = time.time()
    check_interval = 5  # Check every 5 seconds
    
    while time.time() - start_time < SERVER_TIMEOUT:
        # Check if server process is still running
        if server_process.poll() is not None:
            print("❌ Server process terminated unexpectedly")
            print_server_log()
            pytest.fail("Server process terminated unexpectedly")
        
        # Check if server log file exists and has startup complete message
        if Path(SERVER_LOG).exists():
            try:
                with open(SERVER_LOG, 'r') as f:
                    log_content = f.read()
                
                if startup_complete_message in log_content:
                    elapsed_time = time.time() - start_time
                    print(f"✅ Server startup complete! (took {elapsed_time:.1f} seconds)")
                    return True
                    
            except Exception as e:
                print(f"⚠️ Error reading server log: {e}")
        
        # Wait before next check
        elapsed_time = time.time() - start_time
        print(f"⏱️ Still waiting... ({elapsed_time:.1f}/{SERVER_TIMEOUT} seconds)")
        time.sleep(check_interval)
    
    # If we reach here, timeout occurred
    elapsed_time = time.time() - start_time
    print(f"⚠️ Server startup timeout after {elapsed_time:.1f} seconds")
    print("📋 Checking for startup complete message one more time...")
    
    # Final check
    if Path(SERVER_LOG).exists():
        try:
            with open(SERVER_LOG, 'r') as f:
                log_content = f.read()
            if startup_complete_message in log_content:
                print("✅ Found startup complete message in final check!")
                return True
        except Exception:
            pass
    
    # If still no startup complete message, wait additional 30 seconds as fallback
    print("⏳ No startup complete message found, waiting additional 30 seconds as fallback...")
    time.sleep(30)
    print("✅ Fallback wait completed")
    print_server_log()
    return True


def run_benchmark():
    """Run client benchmark"""
    global server_process
    
    # Make sure server is running
    if not server_process or server_process.poll() is not None:
        print("❌ Server is not running or has terminated")
        print_server_log()
        pytest.skip("Server is not running")
    
    print(f"\n🔥 Starting client benchmark...")
    print(f"📊 Prompts: {NUM_PROMPTS}, Concurrency: {MAX_CONCURRENCY}")
    
    client_cmd = [
        # "vllm", "bench", "serve",
        "coverage", "run", "--parallel-mode", "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--model", MODEL_PATH,
        "--num-prompts", str(NUM_PROMPTS),
        "--max-concurrency", str(MAX_CONCURRENCY),
        "--random-input-len", "256",
        "--random-output-len", "512",
        "--trust-remote-code",
        "--ignore_eos"
    ]
    
    print(f"🔧 Client command: {' '.join(client_cmd)}")
    
    try:
        with open(CLIENT_LOG, 'w') as log_file:
            result = subprocess.run(
                client_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
        
        print("✅ Client benchmark completed successfully")
        print(f"📝 Client logs: {CLIENT_LOG}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Client benchmark failed with return code: {e.returncode}")
        print_server_log()
        pytest.fail(f"Client benchmark failed with return code: {e.returncode}")


def validate_performance_metrics():
    """Validate performance metrics from client log"""
    print(f"\n📊 Validating performance metrics...")
    
    # Check if client log exists
    if not Path(CLIENT_LOG).exists():
        pytest.fail(f"Client log file not found: {CLIENT_LOG}")
    
    try:
        with open(CLIENT_LOG, 'r') as f:
            log_content = f.read()
        
        # Extract Output token throughput value using regex
        # Pattern: "Output token throughput (tok/s):         768.97598"
        pattern = r'Output token throughput \(tok/s\):\s+(\d+\.?\d*)'
        match = re.search(pattern, log_content)
        
        if not match:
            pytest.fail("Could not find 'Output token throughput (tok/s):' in client log")
        
        throughput_value = float(match.group(1))
        print(f"📈 Output token throughput: {throughput_value:.2f} tok/s")
        print(f"🎯 Required minimum: {MIN_OUTPUT_TOKEN_THROUGHPUT} tok/s")
        
        # Validate throughput meets minimum requirement
        if throughput_value < MIN_OUTPUT_TOKEN_THROUGHPUT:
            raise AssertionError(
                f"Performance test failed: Output token throughput {throughput_value:.2f} tok/s "
                f"is below minimum requirement of {MIN_OUTPUT_TOKEN_THROUGHPUT} tok/s"
            )
        
        print(f"✅ Performance validation passed! Throughput: {throughput_value:.2f} tok/s (>= {MIN_OUTPUT_TOKEN_THROUGHPUT} tok/s)")
        return throughput_value
        
    except FileNotFoundError:
        pytest.fail(f"Client log file not found: {CLIENT_LOG}")
    except Exception as e:
        pytest.fail(f"Error reading client log: {e}")


def test_complete_pipeline():
    """Test: Complete benchmark pipeline (综合测试)"""
    print("\n🧪 Running complete benchmark pipeline...")
    
    # Install dependencies
    install_dependencies()
    
    # Check model path
    check_model_path_exists()
    
    # Start server
    start_server()
    
    # Wait for server to be ready
    wait_for_server_ready()
    
    # Run benchmark
    run_benchmark()
    
    # Validate performance metrics
    throughput = validate_performance_metrics()
    
    print("\n" + "=" * 60)
    print("📊 Benchmark Results")
    print("=" * 60)
    print(f"Server log: {Path(SERVER_LOG).absolute()}")
    print(f"Client log: {Path(CLIENT_LOG).absolute()}")
    print(f"🚀 Output token throughput: {throughput:.2f} tok/s")
    print("🎉 Complete pipeline test passed!")
