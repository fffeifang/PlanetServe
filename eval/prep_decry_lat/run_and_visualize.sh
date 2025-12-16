#!/bin/bash

echo "===== Running the benchmark ====="
bash run_standalone_eval.sh

if [ ! -f "../../build/model_prepare_performance.log" ]; then
    echo "Error: Log file not found. The benchmark may have failed."
    exit 1
fi

echo "===== Generating CDF plot ====="
python visualize_latency_cdf.py

