#!/usr/bin/env bash
# Bundle every SR-campaign result EXCEPT multi-GB binaries (checkpoints,
# capture tensors) into one dated tar.gz for transfer/archival.
set -euo pipefail
STAMP=$(date -u +%Y%m%dT%H%M)
OUT=${1:-/home/ubuntu/xzz/sr_results_bundle_$STAMP.tar.gz}
P=/home/ubuntu/xzz/of3_perf_runs
tar czf "$OUT" \
  --exclude='*.ckpt' --exclude='case_*.pt' --exclude='*.pt' \
  -C /home/ubuntu/xzz \
  of3_perf_runs/sr_stats_blocks \
  of3_perf_runs/sr_pre400 \
  of3_perf_runs/sr_fork1000 \
  of3_perf_runs/sr_srfull \
  of3_perf_runs/sr_capture_640/capture.log \
  of3_perf_runs/sr_capture_matrix/c640_skip20.log \
  of3_perf_runs/sr_capture_matrix/c640_skip60.log \
  of3_perf_runs/sr_capture_matrix/c384_skip0.log \
  of3_perf_runs/sr_stats_hook \
  2>/dev/null || true
ls -lh "$OUT"
sha256sum "$OUT"
