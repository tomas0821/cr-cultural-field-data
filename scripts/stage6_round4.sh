#!/bin/bash
# Round-3 robustness reruns (2026-10-24): landscape-generated l_c/w profile (misspecification check), free binomial scale h,
# two more full-panel detection-floor replicates (2.5 % slow term; 1 % term). Sequential; ~8 h.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=8
LOG=data/processed/stage6_round4.log; echo "START $(date)" > $LOG
python3 scripts/stage3a_recovery_panel.py --lc 3 --lc_profile --landscape --sets 3 --variants comb_rw --suffix _landscape > data/processed/stage3a_lcprof_landscape.log 2>&1; echo "landscape done $(date)" >> $LOG
python3 scripts/stage6_joint_dynamics.py --models hfree > data/processed/stage6_r4_hfree.log 2>&1; echo "hfree done $(date)" >> $LOG
python3 scripts/stage6_joint_dynamics.py --synthetic SKD2s --models S,SK,SKD --out stage6_recovery_SKD2s.json > data/processed/stage6_r4_recSKD2s.log 2>&1; echo "recSKD2s done $(date)" >> $LOG
python3 scripts/stage6_joint_dynamics.py --synthetic SKD1 --models S,SK,SKD --out stage6_recovery_SKD1.json > data/processed/stage6_r4_recSKD1.log 2>&1; echo "recSKD1 done $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
