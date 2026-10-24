#!/bin/bash
# Round-2 robustness reruns (2026-10-23), sequential. Logs in data/processed/stage6_round3.log and stage6_r3_*.log.
set -u; cd "$(dirname "$0")/.."
P=data/processed; S6=scripts/stage6_joint_dynamics.py
echo "START $(date)" > $P/stage6_round3.log
python3 scripts/stage3a_recovery_panel.py --lc 3   --lc_profile --sets 2 --variants comb_rw --sigma_mu2 0.045 --suffix _lc3  > $P/stage3a_lcprof3.log  2>&1; echo "lcprof3 done $(date)"  >> $P/stage6_round3.log
python3 scripts/stage3a_recovery_panel.py --lc 4.5 --lc_profile --sets 2 --variants comb_rw --sigma_mu2 0.045 --suffix _lc45 > $P/stage3a_lcprof45.log 2>&1; echo "lcprof45 done $(date)" >> $P/stage6_round3.log
python3 $S6 --synthetic SK --subsample 1000 --models SKprof,SKprof2 --out stage6_recovery_SK.json > $P/stage6_r3_synkprof.log 2>&1; echo "synkprof done $(date)" >> $P/stage6_round3.log
python3 $S6 --models SKU,SKUD,SKr50                                  > $P/stage6_r3_sku.log      2>&1; echo "sku done $(date)"      >> $P/stage6_round3.log
python3 $S6 --synthetic SKD2 --models SK,SKD --out stage6_recovery_SKD2.json > $P/stage6_r3_recSKD2.log 2>&1; echo "recSKD2 done $(date)" >> $P/stage6_round3.log
python3 $S6 --lc 3   --models SD,SK,SKD --out stage6_lc3.json       > $P/stage6_r3_lc3.log      2>&1; echo "lc3 done $(date)"      >> $P/stage6_round3.log
python3 $S6 --lc 4.5 --models SD,SK,SKD --out stage6_lc45.json      > $P/stage6_r3_lc45.log     2>&1; echo "lc45 done $(date)"     >> $P/stage6_round3.log
python3 $S6 --h 1.1  --models SK --out stage6_h11.json              > $P/stage6_r3_h11.log      2>&1; echo "h11 done $(date)"      >> $P/stage6_round3.log
python3 $S6 --h 1.76 --models SK --out stage6_h176.json             > $P/stage6_r3_h176.log     2>&1; echo "h176 done $(date)"     >> $P/stage6_round3.log
python3 scripts/stage3_gff_profile.py --lcs 7,10 --ks 25,43 --suffix _matchedk > $P/stage3_matchedk.log 2>&1; echo "matchedk done $(date)" >> $P/stage6_round3.log
echo "ALL DONE $(date)" >> $P/stage6_round3.log
