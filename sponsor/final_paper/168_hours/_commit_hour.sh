#!/bin/bash
# Generate, format, commit and push files for a single hour.
# Usage: ./_commit_hour.sh <global_hour>
set -e

HOUR=$1
DAY=$(( HOUR / 24 + 1 ))
HOD=$(( HOUR % 24 ))
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
GEN_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Hour ${HOUR} (Day ${DAY}, H${HOD}) ==="

# Generate files
cd "$GEN_DIR"
python3 _gen_hourly.py "$HOUR"

# Format Python files
cd "$REPO_ROOT"
DAY_DIR="sponsor/final_paper/168_hours/day_$(printf '%02d' $DAY)"
ruff format "${DAY_DIR}/hourly/sponsor_hour_$(printf '%03d' $HOUR).py" 2>/dev/null || true
ruff check --fix "${DAY_DIR}/hourly/sponsor_hour_$(printf '%03d' $HOUR).py" 2>/dev/null || true

# Stage files
git add "${DAY_DIR}/hourly/sponsor_hour_$(printf '%03d' $HOUR).py"
git add "${DAY_DIR}/hourly/output/sponsor_hour_$(printf '%03d' $HOUR)_output.json"
git add "${DAY_DIR}/diagrams/sponsor_decision_flow_hour_$(printf '%03d' $HOUR).txt"
git add "${DAY_DIR}/diagrams/agent_workload_hour_$(printf '%03d' $HOUR).txt"
git add "${DAY_DIR}/diagrams/robot_auth_timeline_hour_$(printf '%03d' $HOUR).txt"

# Commit
PERIOD_DESC=""
case $HOD in
    0|1|2) PERIOD_DESC="overnight low-volume";;
    3|4|5) PERIOD_DESC="early morning";;
    6) PERIOD_DESC="morning ramp-up";;
    7|8) PERIOD_DESC="morning peak";;
    9|10|11) PERIOD_DESC="peak operations";;
    12) PERIOD_DESC="midday steady";;
    13|14) PERIOD_DESC="afternoon ops";;
    15) PERIOD_DESC="afternoon steady";;
    16) PERIOD_DESC="afternoon wind-down";;
    17) PERIOD_DESC="evening transition";;
    18) PERIOD_DESC="evening ops";;
    19) PERIOD_DESC="evening wind-down";;
    20) PERIOD_DESC="night transition";;
    21) PERIOD_DESC="night ops";;
    22|23) PERIOD_DESC="overnight transition";;
esac

git commit -m "$(cat <<EOF
feat(168h): Day ${DAY} Hour ${HOD} - ${PERIOD_DESC} sponsor activity (H${HOUR})

- sponsor_hour_$(printf '%03d' $HOUR).py with 12 decisions at 5-min intervals
- JSON output with patient arrivals, robot status, and decision records
- 3 text diagrams: decision flow, agent workload, robot authorization

https://claude.ai/code/session_01EQVtPbMxjDi4CbGxvssVc4
EOF
)"

# Push with retry
for i in 1 2 3 4; do
    if git push -u origin "$(git branch --show-current)" 2>/dev/null; then
        echo "=== Hour ${HOUR} committed and pushed ==="
        exit 0
    fi
    DELAY=$((2 ** i))
    echo "Push failed, retrying in ${DELAY}s..."
    sleep $DELAY
done
echo "ERROR: Push failed after 4 retries"
exit 1
