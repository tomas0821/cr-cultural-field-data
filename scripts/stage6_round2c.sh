#!/bin/bash
# Round-1 follow-up (2026-10-23): tighter rate brackets (SKprof2); waits for stage6_round2b.sh (ewprof) to finish.
set -u; cd "$(dirname "$0")/.."
P=data/processed
until grep -q "ewprof done" $P/stage6_round2.log; do sleep 120; done
python3 scripts/stage6_joint_dynamics.py --models SKprof2 > $P/stage6_r2_kprof2.log 2>&1; echo "kprof2 done $(date)" >> $P/stage6_round2.log
