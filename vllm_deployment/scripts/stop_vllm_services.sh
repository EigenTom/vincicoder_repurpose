#!/bin/bash
# Script to stop vLLM services and load balancer (standalone deployment)
# Usage: ./scripts/stop_vllm_services.sh

# Don't use set -e here because we want to continue even if some kills fail
# set -e

# Deployment root: parent of scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DEPLOY_ROOT"

LOG_DIR="$DEPLOY_ROOT/logs/vllm"

# Function to recursively collect all descendant PIDs of a given PID
# Returns space-separated list of all PIDs in the process tree
collect_descendant_pids() {
    local root_pid=$1
    local all_pids="$root_pid"
    local to_check="$root_pid"
    local new_pids=""

    # Iteratively collect all descendants until no new children are found
    while [ -n "$to_check" ]; do
        new_pids=""
        for pid in $to_check; do
            # Find direct children of this PID
            local children=$(ps -o pid --no-headers --ppid $pid 2>/dev/null | tr '\n' ' ')
            if [ -n "$children" ]; then
                # Add children to the list
                all_pids="$all_pids $children"
                new_pids="$new_pids $children"
            fi
        done
        to_check="$new_pids"
    done

    echo "$all_pids"
}

# Function to collect log files opened by a process and its descendants
# Returns newline-separated list of log file paths
collect_process_log_files() {
    local root_pid=$1
    local log_files=""

    # Collect all descendant PIDs (including the root)
    local all_pids=$(collect_descendant_pids $root_pid)

    # Use lsof to find all log files opened by these processes
    for pid in $all_pids; do
        [ -z "$pid" ] && continue
        if ps -p $pid > /dev/null 2>&1; then
            local log_dir_abs=$(cd "$DEPLOY_ROOT" && cd "$LOG_DIR" 2>/dev/null && pwd || echo "$LOG_DIR")
            local pid_logs=$(lsof -p $pid 2>/dev/null | awk 'NR>1 {print $NF}' | grep -E "\.log$" | grep -E "(^$log_dir_abs/|$LOG_DIR/)" | sort -u)
            if [ -n "$pid_logs" ]; then
                log_files="$log_files"$'\n'"$pid_logs"
            fi
        fi
    done

    echo "$log_files" | grep -v '^$' | sort -u
}

# Function to kill a PID and all its descendants
kill_process_tree() {
    local root_pid=$1
    local signal=${2:-TERM}

    if ! ps -p $root_pid > /dev/null 2>&1; then
        return 1
    fi

    local all_pids=$(collect_descendant_pids $root_pid)

    if [ "$signal" = "KILL" ]; then
        for pid in $all_pids; do
            [ -z "$pid" ] && continue
            kill -KILL $pid 2>/dev/null || true
        done
    else
        local pids_array=($all_pids)
        for ((idx=${#pids_array[@]}-1; idx>=0; idx--)); do
            pid=${pids_array[$idx]}
            [ -z "$pid" ] && continue
            kill -TERM $pid 2>/dev/null || true
        done
    fi
}

echo "=== Stopping vLLM Services and Load Balancer ==="
echo "Deployment root: $DEPLOY_ROOT"
echo ""

# Step 1: Stop load balancer (if PID file exists)
echo "Step 1: Stopping vLLM load balancer..."
LB_PID_FILE="$LOG_DIR/vllm_lb_pid.txt"
LB_LOG_FILES=""
if [ -f "$LB_PID_FILE" ]; then
    LB_PID=$(cat "$LB_PID_FILE" 2>/dev/null | head -1)
    if [ -n "$LB_PID" ] && ps -p $LB_PID > /dev/null 2>&1; then
        echo "  Killing load balancer PID $LB_PID..."
        LB_LOG_FILES=$(collect_process_log_files $LB_PID)
        if [ -z "$LB_LOG_FILES" ]; then
            LB_LOG_FILES=$(find "$LOG_DIR" -maxdepth 1 -name "load_balancer*.log" -type f 2>/dev/null)
        fi
        kill -TERM $LB_PID 2>/dev/null || true
        sleep 2
        if ps -p $LB_PID > /dev/null 2>&1; then
            echo "  Force killing load balancer PID $LB_PID..."
            kill -KILL $LB_PID 2>/dev/null || true
        fi
        echo "  Load balancer stopped"
        rm -f "$LB_PID_FILE"

        if [ -n "$LB_LOG_FILES" ]; then
            echo "  Removing load balancer log files..."
            while IFS= read -r log_file; do
                [ -z "$log_file" ] && continue
                if [ -f "$log_file" ]; then
                    rm -f "$log_file"
                    echo "    Removed: $log_file"
                fi
            done <<< "$LB_LOG_FILES"
        else
            echo "  Note: Could not detect load balancer log file (process may have already terminated)"
        fi
    else
        echo "  Load balancer PID from file not found (may have already terminated)"
        rm -f "$LB_PID_FILE"
    fi
else
    echo "  No load balancer PID file found ($LB_PID_FILE)"
    echo "  If load balancer is running, you may need to find and kill it manually"
fi

echo ""

# Step 2: Stop vLLM services (ONLY the ones we started)
echo "Step 2: Stopping vLLM services (only the ones we started)..."

if [ -f "$LOG_DIR/vllm_pids.txt" ]; then
    echo "  Reading PIDs from $LOG_DIR/vllm_pids.txt"

    pids_array=()
    while IFS= read -r pid || [ -n "$pid" ]; do
        [ -z "$pid" ] && continue
        pids_array+=($pid)
    done < "$LOG_DIR/vllm_pids.txt"

    KILLED_COUNT=0
    NOT_FOUND_COUNT=0

    vllm_log_files=""
    for pid in "${pids_array[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            pid_logs=$(collect_process_log_files $pid)
            if [ -n "$pid_logs" ]; then
                vllm_log_files="$vllm_log_files"$'\n'"$pid_logs"
            fi
        fi
    done

    for pid in "${pids_array[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Killing vLLM service PID $pid and all its descendant processes..."
            descendant_pids=$(collect_descendant_pids $pid)
            pid_count=$(echo $descendant_pids | wc -w)
            echo "    Found $pid_count process(es) in the process tree"
            kill_process_tree $pid TERM
            KILLED_COUNT=$((KILLED_COUNT + 1))
        else
            echo "  PID $pid: Process not found (may have already terminated)"
            NOT_FOUND_COUNT=$((NOT_FOUND_COUNT + 1))
        fi
    done

    if [ $KILLED_COUNT -gt 0 ]; then
        echo "  Waiting 3 seconds for graceful shutdown..."
        sleep 3

        for pid in "${pids_array[@]}"; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "  Force killing vLLM service PID $pid and all its descendant processes..."
                descendant_pids=$(collect_descendant_pids $pid)
                pid_count=$(echo $descendant_pids | wc -w)
                echo "    Force killing $pid_count process(es) in the process tree"
                kill_process_tree $pid KILL
            fi
        done

        echo "  Stopped $KILLED_COUNT vLLM service(s)"
    else
        echo "  No running processes found from saved PIDs"
    fi

    if [ $NOT_FOUND_COUNT -gt 0 ]; then
        echo "  ($NOT_FOUND_COUNT process(es) were already terminated)"
    fi

    if [ -n "$vllm_log_files" ]; then
        echo ""
        echo "  Removing vLLM service log files..."
        removed_count=0
        while IFS= read -r log_file; do
            [ -z "$log_file" ] && continue
            if [ -f "$log_file" ]; then
                rm -f "$log_file"
                echo "    Removed: $log_file"
                removed_count=$((removed_count + 1))
            fi
        done <<< "$vllm_log_files"
        if [ $removed_count -eq 0 ] && [ -n "$vllm_log_files" ]; then
            echo "    (No log files found to remove - they may have already been deleted)"
        fi
    else
        echo ""
        echo "  Note: Could not detect vLLM log files (processes may have already terminated)"
    fi

    rm -f "$LOG_DIR/vllm_pids.txt"
else
    echo "  WARNING: $LOG_DIR/vllm_pids.txt not found!"
    echo "  Cannot safely stop services without PID file."
    echo "  If you know the PIDs, you can manually kill them."
    echo "  To avoid affecting other users, DO NOT use pkill!"
fi

echo ""
echo "  NOTE: Only processes from vllm_pids.txt were killed."
echo "  Other vLLM services (if any) were NOT affected."

echo ""
echo "=== Checking GPU status ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E '^ 4,|^ 5,|^ 6,|^ 7,' || echo "GPU 4,5,6,7 status:"

echo ""
echo "Done!"
