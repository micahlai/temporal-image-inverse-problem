#!/usr/bin/env bash
set -euo pipefail
implementation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda run --no-capture-output -n temporal-torch python "$implementation_dir/main.py" \
  "$implementation_dir/../og-cpu-implementation/data/pure_conv/dataset.npz" \
  --all-priors true --lambda-spatial -7 --lambda-temporal -7 --num-iters 250 \
  --output-dir "$implementation_dir/results/pure_conv_lo-lambda" "$@"
