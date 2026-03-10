#!/bin/bash

LOG_FILE="/tmp/efsmi.log"
PID_FILE="/tmp/monitor_gcu.pid"
LOCK_FILE="/tmp/monitor_gcu.lock"  # File lock for mutual exclusion
CHECK_INTERVAL=5
EXEC_INTERVAL=10
TIMEOUT=180
TARGET_PROCS=("EngineCore" "vllm" "pytest")

# Log rotation settings
MAX_LOG_SIZE=$((10 * 1024 * 1024))
MAX_LOG_BACKUPS=5

# ============================================================================
# Enhanced PID Management Functions
# ============================================================================

# Check if another instance is already running
check_existing_instance() {
    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE" 2>/dev/null)
        
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            # Process exists, verify it's actually our script
            if ps -p "$old_pid" -o cmd= 2>/dev/null | grep -q "monitor_gcu.sh"; then
                echo "ERROR: Another monitor instance is already running (PID: $old_pid)"
                echo "Use './stop_monitor.sh' or 'kill $old_pid' to stop it first"
                exit 1
            else
                # PID exists but not our script, clean up stale PID file
                echo "WARNING: Stale PID file found, cleaning up..."
                rm -f "$PID_FILE"
            fi
        else
            # PID file exists but process is dead
            echo "WARNING: Removing stale PID file (process $old_pid not found)"
            rm -f "$PID_FILE"
        fi
    fi
}

# Create PID file with validation
create_pid_file() {
    # Use file lock to ensure atomic operation
    (
        flock -n 200 || {
            echo "ERROR: Cannot acquire lock, another instance may be starting"
            exit 1
        }
        
        # Write PID and additional metadata
        cat > "$PID_FILE" << EOF
$$
$(date +%s)
$(readlink -f "$0")
EOF
        
    ) 200>"$LOCK_FILE"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create PID file"
        exit 1
    fi
}

# Validate PID file integrity
validate_pid_file() {
    if [ ! -f "$PID_FILE" ]; then
        log_message "CRITICAL: PID file disappeared! This process may be orphaned."
        log_message "Current PID: $$, can be stopped manually with: kill $$"
        return 1
    fi
    
    local recorded_pid=$(head -n 1 "$PID_FILE" 2>/dev/null)
    if [ "$recorded_pid" != "$$" ]; then
        log_message "CRITICAL: PID mismatch! Expected $$, found $recorded_pid"
        log_message "PID file may have been overwritten by another instance"
        return 1
    fi
    
    return 0
}

# ============================================================================
# Log Rotation Functions
# ============================================================================

rotate_log() {
    if [ ! -f "$LOG_FILE" ]; then
        return 0
    fi
    
    local log_size=$(stat -c%s "$LOG_FILE" 2>/dev/null || stat -f%z "$LOG_FILE" 2>/dev/null)
    
    if [ "$log_size" -ge "$MAX_LOG_SIZE" ]; then
        local timestamp=$(date '+%Y%m%d_%H%M%S')
        local archive_file="${LOG_FILE}.${timestamp}"
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log rotation: $LOG_FILE → $archive_file (size: $log_size bytes)" | tee -a "$LOG_FILE"
        
        mv "$LOG_FILE" "$archive_file"
        gzip "$archive_file" 2>/dev/null && archive_file="${archive_file}.gz"
        touch "$LOG_FILE"
        
        local backup_count=$(ls -1t "${LOG_FILE}".* 2>/dev/null | wc -l)
        if [ "$backup_count" -gt "$MAX_LOG_BACKUPS" ]; then
            ls -1t "${LOG_FILE}".* | tail -n +$((MAX_LOG_BACKUPS + 1)) | xargs rm -f 2>/dev/null
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaned up old log backups (kept last $MAX_LOG_BACKUPS)" >> "$LOG_FILE"
        fi
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log rotation completed" >> "$LOG_FILE"
    fi
}

log_message() {
    rotate_log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# Monitor Functions
# ============================================================================

check_process() {
    for proc in "${TARGET_PROCS[@]}"; do
        if pgrep -x "$proc" > /dev/null 2>&1 || pgrep -f "$proc" > /dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

run_efsmi() {
    # Validate PID file before expensive operation
    if ! validate_pid_file; then
        log_message "WARNING: Continuing despite PID file issue..."
    fi
    
    rotate_log
    log_message "Target process detected, executing efsmi"
    if command -v efsmi &> /dev/null; then
        efsmi --query >> "$LOG_FILE" 2>&1
        log_message "efsmi completed with exit code: $?"
    else
        log_message "ERROR: efsmi command not found"
    fi
    echo "**************************************************" >> "$LOG_FILE"
}

cleanup() {
    log_message "Monitor script exiting (PID: $$)"
    
    # Verify we own the PID file before deleting
    if [ -f "$PID_FILE" ]; then
        local recorded_pid=$(head -n 1 "$PID_FILE" 2>/dev/null)
        if [ "$recorded_pid" = "$$" ]; then
            rm -f "$PID_FILE"
            log_message "PID file cleaned up successfully"
        else
            log_message "WARNING: Not removing PID file (owned by PID $recorded_pid, not $$)"
        fi
    else
        log_message "WARNING: PID file already removed"
    fi
    
    # Clean up lock file
    rm -f "$LOCK_FILE"
    
    exit 0
}

# ============================================================================
# Main Execution
# ============================================================================

# Trap signals for graceful shutdown
trap cleanup SIGINT SIGTERM

# Check for existing instance
check_existing_instance

# Create PID file with lock
create_pid_file

log_message "================================================"
log_message "GPU monitor started successfully"
log_message "PID: $$"
log_message "Script: $(readlink -f "$0")"
log_message "Watching processes: ${TARGET_PROCS[*]}"
log_message "Log file: $LOG_FILE"
log_message "Log rotation: enabled (max size: $((MAX_LOG_SIZE / 1024 / 1024))MB, keep last $MAX_LOG_BACKUPS backups)"
log_message "Check interval: ${CHECK_INTERVAL}s, Exec interval: ${EXEC_INTERVAL}s, Timeout: ${TIMEOUT}s"
log_message "================================================"

LAST_EXEC_TIME=0
LAST_SEEN_TIME=$(date +%s)
MONITORING=false
PID_CHECK_COUNTER=0

while true; do
    CURRENT_TIME=$(date +%s)
    
    # Periodically validate PID file (every 10 iterations)
    ((PID_CHECK_COUNTER++))
    if [ $((PID_CHECK_COUNTER % 10)) -eq 0 ]; then
        if ! validate_pid_file; then
            log_message "CRITICAL: PID file integrity check failed, exiting to prevent orphan process"
            # Don't call cleanup since PID file is already gone or corrupted
            rm -f "$LOCK_FILE"
            exit 1
        fi
    fi

    if check_process; then
        LAST_SEEN_TIME=$CURRENT_TIME

        if [ "$MONITORING" = false ]; then
            log_message "Target process detected, starting efsmi monitoring"
            MONITORING=true
        fi

        if [ $((CURRENT_TIME - LAST_EXEC_TIME)) -ge $EXEC_INTERVAL ]; then
            run_efsmi
            LAST_EXEC_TIME=$CURRENT_TIME
        fi
    else
        if [ "$MONITORING" = true ]; then
            TIME_SINCE_LAST_SEEN=$((CURRENT_TIME - LAST_SEEN_TIME))

            if [ $TIME_SINCE_LAST_SEEN -ge $TIMEOUT ]; then
                log_message "No target process detected for ${TIMEOUT}s (3 minutes), exiting"
                cleanup
            else
                REMAINING=$((TIMEOUT - TIME_SINCE_LAST_SEEN))
                log_message "Target process lost, waiting... (${REMAINING}s until timeout)"
            fi
        fi
    fi

    sleep $CHECK_INTERVAL
done
