#!/bin/bash
# Round-1 follow-up (2026-10-23): profile in D for the EW-consistent weight form; waits for stage6_round2.sh to finish.
set -u; cd "$(dirname "$0")/.."
P=data/processed
until grep -q "ALL DONE" $P/stage6_round2.log; do sleep 120; done
echo "START ewprof $(date)" >> $P/stage6_round2.log
python3 scripts/stage6_joint_dynamics.py --models EWprof > $P/stage6_r2_ewprof.log 2>&1; echo "ewprof done $(date)" >> $P/stage6_round2.log
