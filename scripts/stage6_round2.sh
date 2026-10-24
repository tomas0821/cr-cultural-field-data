#!/bin/bash
# Round-1 robustness reruns (2026-10-23), sequential (each step uses all threads). Logs in data/processed/.
set -u; cd "$(dirname "$0")/.."
P=data/processed
echo "START $(date)" > $P/stage6_round2.log
python3 scripts/stage6_joint_dynamics.py --models SKf,SKDf            > $P/stage6_r2_fsmu.log  2>&1; echo "fsmu done $(date)"  >> $P/stage6_round2.log
python3 scripts/stage6_joint_dynamics.py --models SDe,SKDe            > $P/stage6_r2_ew.log    2>&1; echo "ew done $(date)"    >> $P/stage6_round2.log
python3 scripts/stage6_joint_dynamics.py --models SKprof              > $P/stage6_r2_kprof.log 2>&1; echo "kprof done $(date)" >> $P/stage6_round2.log
python3 scripts/stage6_joint_dynamics.py --synthetic SK   --subsample 1000 --models S,SD,SK,SKD --out stage6_recovery_SK.json   > $P/stage6_r2_recSK.log   2>&1; echo "recSK done $(date)"   >> $P/stage6_round2.log
python3 scripts/stage6_joint_dynamics.py --synthetic SKD5 --subsample 1000 --models S,SD,SK,SKD --out stage6_recovery_SKD5.json > $P/stage6_r2_recSKD5.log 2>&1; echo "recSKD5 done $(date)" >> $P/stage6_round2.log
python3 scripts/stage3a_recovery_panel.py --lc 1.5 --k 13 --variants comb_rw --sigma_mu2 0.045 --suffix _k13 > $P/stage3a_k13.log 2>&1; echo "stage3a k13 done $(date)" >> $P/stage6_round2.log
echo "ALL DONE $(date)" >> $P/stage6_round2.log
