#!/usr/bin/env bash
#
# Smoke test for scripts/dev.sh.
#
# Starts dev.sh in the background with streamable-http transport, polls the
# structured readiness endpoint, and asserts:
#   1. GET /health/ready returns HTTP 200
#   2. Response JSON has "ready": true
#   3. Response JSON has tools >= MIN_TOOLS
#
# Fails fast if readiness is not reached within READY_BUDGET_SECONDS.
# Always tears down the backend process on exit.
#
# Usage:
#   ./scripts/smoke_test_dev.sh                   # defaults
#   MIN_TOOLS=20 READY_BUDGET_SECONDS=90 ./scripts/smoke_test_dev.sh
#
# Designed for CI: exits 0 on success, non-zero on any failure.

set -euo pipefail

MIN_TOOLS=${MIN_TOOLS:-10}
READY_BUDGET_SECONDS=${READY_BUDGET_SECONDS:-90}
PORT=${MAVERICK_PORT:-8003}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[smoke]${NC} $*"; }
ok()   { echo -e "${GREEN}[smoke] PASS${NC} $*"; }
fail() { echo -e "${RED}[smoke] FAIL${NC} $*"; }

DEV_PID=""
cleanup() {
    # dev.sh installs its own EXIT trap that kills the backend when dev.sh
    # receives a signal. We signal dev.sh; it forwards to children. The
    # lsof fallback below handles any process that outlives its parent.
    if [ -n "$DEV_PID" ] && kill -0 "$DEV_PID" 2>/dev/null; then
        log "Tearing down dev.sh (pid=$DEV_PID)"
        kill -TERM "$DEV_PID" 2>/dev/null || true
        # Also signal any children of dev.sh (backend uvicorn process)
        if command -v pkill >/dev/null 2>&1; then
            pkill -TERM -P "$DEV_PID" 2>/dev/null || true
        fi
        sleep 2
        kill -KILL "$DEV_PID" 2>/dev/null || true
    fi
    # Belt-and-suspenders: any process still holding the port
    if command -v lsof >/dev/null 2>&1; then
        STALE=$(lsof -ti:"$PORT" 2>/dev/null || true)
        if [ -n "$STALE" ]; then
            log "Killing stale process on port $PORT: $STALE"
            kill -9 $STALE 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT INT TERM

# Launch dev.sh as a background child. Portable — no setsid (not available
# on macOS). Cleanup uses pkill -P to reach grandchildren.
log "Launching ./scripts/dev.sh (transport=streamable-http, port=$PORT)"
./scripts/dev.sh </dev/null >smoke_dev.log 2>&1 &
DEV_PID=$!
log "dev.sh PID: $DEV_PID (logs -> smoke_dev.log)"

READY=false
TOOLS_FOUND=0
START_TS=$(date +%s)
DEADLINE=$((START_TS + READY_BUDGET_SECONDS))

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if ! kill -0 "$DEV_PID" 2>/dev/null; then
        fail "dev.sh exited before becoming ready"
        echo "----- smoke_dev.log (tail) -----"
        tail -n 80 smoke_dev.log || true
        echo "--------------------------------"
        exit 2
    fi

    RESPONSE=$(curl -fsS --max-time 2 "http://localhost:${PORT}/health/ready" 2>/dev/null || true)
    if [ -n "$RESPONSE" ]; then
        # Parse JSON via Python. No f-strings — keeps single-quoted shell
        # argument free of nested double-quote escaping.
        PARSED=$(printf '%s' "$RESPONSE" | python3 -c 'import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(str(d.get("ready", False)).lower(), d.get("tools", 0))
except Exception:
    print("false 0")
' 2>/dev/null || echo "false 0")
        READY_FLAG=$(echo "$PARSED" | awk '{print $1}')
        TOOLS_FOUND=$(echo "$PARSED" | awk '{print $2}')
        TOOLS_FOUND=${TOOLS_FOUND:-0}
        if [ "$READY_FLAG" = "true" ] && [ "$TOOLS_FOUND" -ge "$MIN_TOOLS" ]; then
            READY=true
            break
        elif [ "$READY_FLAG" = "true" ]; then
            log "Ready but tools=$TOOLS_FOUND below MIN_TOOLS=$MIN_TOOLS"
        fi
    fi

    ELAPSED=$(( $(date +%s) - START_TS ))
    log "Waiting for readiness... elapsed=${ELAPSED}s / ${READY_BUDGET_SECONDS}s"
    sleep 2
done

if [ "$READY" != true ]; then
    fail "Readiness not reached within ${READY_BUDGET_SECONDS}s (tools=$TOOLS_FOUND, need>=$MIN_TOOLS)"
    echo "----- smoke_dev.log (tail) -----"
    tail -n 80 smoke_dev.log || true
    echo "--------------------------------"
    echo "----- backend.log (tail) -------"
    tail -n 80 backend.log 2>/dev/null || true
    echo "--------------------------------"
    exit 1
fi

ELAPSED=$(( $(date +%s) - START_TS ))
ok "Backend ready in ${ELAPSED}s with tools=$TOOLS_FOUND (>= $MIN_TOOLS required)"
exit 0
