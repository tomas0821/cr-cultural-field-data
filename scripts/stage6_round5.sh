#!/bin/bash
# Round-4 confound check (2026-10-24): district-level social covariates (MIDEPLAN IDS 2023 + 2011 census schooling) in the mean.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=8
LOG=data/processed/stage6_round5.log; echo "START $(date)" > $LOG
python3 scripts/stage4d_model_comparison.py --social > data/processed/stage4d_social.log 2>&1; echo "stage4d social done $(date)" >> $LOG
python3 scripts/stage6_joint_dynamics.py --social --models S,SK,SKU --out stage6_social.json > data/processed/stage6_r5_social.log 2>&1; echo "stage6 social done $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
