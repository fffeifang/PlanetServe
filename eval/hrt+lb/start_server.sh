for X in {0..7}
do
  echo "[INFO] Launching vLLM instance mnode$X on GPU $X (port 800$X)..."
  CUDA_VISIBLE_DEVICES=$X python -m vllm.entrypoints.openai.api_server \
    --model ../../models/llama-3.1-8b-instruct \
    --served-model-name mnode$X \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --max-num-seqs 4 \
    --port 800$X > logs/llama3_$X.log 2>&1 &

done


