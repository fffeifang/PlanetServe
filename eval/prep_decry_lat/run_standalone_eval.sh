#!/bin/bash

set -e  

echo "===== Building the standalone model preparation evaluation ====="
cd ../..
mkdir -p build
cd build
cmake ..
make test_model_prepare_standalone

echo "===== Running the model preparation evaluation with real Llama model ====="
./test_model_prepare_standalone

echo "===== Evaluation complete ====="
echo "Results saved to model_prepare_performance.log"
echo "See the log file for detailed performance metrics."

echo "===== Summary ====="
tail -n 10 model_prepare_performance.log 