#!/bin/bash
# Generate the long-horizon and fine-dt Navier-Stokes truth, gate it on initial energy,
# and evaluate the checkpoints on it.
#
# Paper: Appendix A.9, Tables A.6 and A.7.
#
# Usage (from the repository root; ~2.2 h on 4 GPUs, set GPUS / PY to override):
#   bash long_horizon/run_pipeline.sh
#
# 4 x 50 trajectories per file. Energy gate: mean omega^2 at t = 1 must lie within
# [0.90, 1.10] of the benchmark test split (needs data/ns/NavierStokes_V1e-5_N1200_T20.mat);
# a file that fails is not evaluated. The paper's values: long_horizon/results/*_gate.json.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPUS=(${GPUS:-0 1 2 3})
O=outputs/long_horizon
mkdir -p $O/parts $O/results/long $O/results/finedt $O/logs

LGN=checkpoints/ns/lgnkm_nu1e5_s0.pt
FNO=checkpoints/ns/fno_nu1e5.pt
LRAN=checkpoints/baselines/lran_nu1e5_s0.pt
KNO=checkpoints/baselines/kno_nu1e5_s0.pt

long_pids=(); fd_pids=(); long_parts=(); fd_parts=()
for i in 0 1 2 3; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  s=$((1234 + i)); f=$O/parts/long_s$s.mat; long_parts+=($f)
  CUDA_VISIBLE_DEVICES=$g $PY -u long_horizon/generate_ns_long.py --N 50 --T 200 --seed $s --out $f \
    > $O/logs/gen_long_s$s.log 2>&1 &
  long_pids+=($!)
  s=$((4321 + i)); f=$O/parts/finedt_s$s.mat; fd_parts+=($f)
  CUDA_VISIBLE_DEVICES=$g $PY -u long_horizon/generate_ns_finedt.py --N 50 --seed $s --out $f \
    > $O/logs/gen_finedt_s$s.log 2>&1 &
  fd_pids+=($!)
done
echo "$(date '+%F %T') launched long ${long_pids[*]} finedt ${fd_pids[*]}"

fail=0
for p in "${fd_pids[@]}"; do wait $p || { echo "finedt generator $p failed"; fail=1; }; done
FD=$O/ns_finedt_nu1e5_N200_T20_fixed.mat
if [ $fail = 0 ] && $PY long_horizon/concat_and_gate.py $FD 3 "${fd_parts[@]}" > $O/logs/gate_finedt.log 2>&1; then
  echo "$(date '+%F %T') finedt gate PASSED; evaluating"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY -u long_horizon/eval_fine_dt.py --data $FD --koopman_ckpt $LGN \
    --save_dir $O/results/finedt > $O/logs/eval_finedt.log 2>&1 && echo "finedt eval done"
else
  echo "$(date '+%F %T') finedt STOPPED (generator failure or gate); see $O/logs/gate_finedt.log"
fi

fail=0
for p in "${long_pids[@]}"; do wait $p || { echo "long generator $p failed"; fail=1; }; done
LONG=$O/ns_long_nu1e5_N200_T200_fixed.mat
if [ $fail = 0 ] && $PY long_horizon/concat_and_gate.py $LONG 0 "${long_parts[@]}" > $O/logs/gate_long.log 2>&1; then
  echo "$(date '+%F %T') long gate PASSED; evaluating"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY -u long_horizon/eval_long_horizon.py --data $LONG --koopman_ckpt $LGN \
    --fno_ckpt $FNO --lran_ckpt $LRAN --kno_ckpt $KNO \
    --save_dir $O/results/long > $O/logs/eval_long.log 2>&1 && echo "long eval done"
else
  echo "$(date '+%F %T') long STOPPED (generator failure or gate); see $O/logs/gate_long.log"
fi
echo "$(date '+%F %T') pipeline finished; compare $O/results with long_horizon/results"
