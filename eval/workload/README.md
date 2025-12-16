# PlanetServe Workload Measurer
---

## What’s in this folder

### 1) Online workload driver + dispatcher (scheduling and load balancing prototype)
An online workload driver that issues OpenAI-compatible requests to model servers under Poisson inter-arrival times. The dispatcher supports:

- `--dispatch lb`: a **prefix-affinity + load-aware** policy aligned with the paper’s HR-tree / load-balancing scheduling.
- `--dispatch rr`: a **round-robin baseline**.

### 2) System-level monitoring from vLLM `/metrics`
During experiments, the tool polls each vLLM server’s `/metrics` endpoint to monitor:
- KV cache usage
- GPU prefix cache hit ratio
- cumulative TTFT and TPOT averages
- token throughput and request throughput

This setup is intended to **observe and quantify serving-side behavior** under real-world load and different load behabvior.

---
## Prerequisites

- **Ubuntu 24.04 LTS** (recommended).
- **8× NVIDIA GPUs** (a quickstart can start from 2 GPUs ; at least A6000 48GB recommended but not strictly required for the quickstart if you use a smaller model).
- Recent NVIDIA software stack (driver/CUDA/cuDNN/NCCL).  
  For convenience, we recommend installing via **Lambda Stack 24.04**:  
  https://lambda.ai/lambda-stack-deep-learning-software
- A **vLLM OpenAI-compatible server** that exposes Prometheus metrics at `/metrics`.

---
# Getting Started

These steps can starts from running a small **two-node** “kick-the-tires” experiment to validate end-to-end functionality.

## Step 1 — Create a Python environment and install vLLM

We use a Conda environment for isolation and install vLLM from source.

```bash
sudo apt update

conda create -n vllm python=3.10 -y
conda activate vllm

curl -LsSf https://astral.sh/uv/install.sh | sh
pip install -U "huggingface_hub[cli]"
uv pip install matplotlib openai

cd deps/vllm
# Use precompiled components when available
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

## Step 2 — Download model weights
```bash
huggingface-cli login
hf download meta-llama/Llama-3.1-8B-Instruct
```

## Step 3 — Start vLLM model servers
### Terminal 0 (Node 0, Port 8000)
```bash
conda activate vllm0
export CUDA_VISIBLE_DEVICES=0

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --dtype auto \
  --api-key token-abc123 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95
```
### Terminal 1 (Node 1, Port 8001)
```bash
conda activate vllm1
export CUDA_VISIBLE_DEVICES=1

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8001 \
  --dtype auto \
  --api-key token-abc123 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95
```
### Terminal n (Node n, Port #Your Port#)
```bash
conda activate vllmn
export CUDA_VISIBLE_DEVICES=n

vllm serve #Your model id# \
  --port #Your Port# \
  --dtype auto \
  --api-key #Your Key# \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95
```
## Step 4 — Run the workload measurer - Quick start
```bash
python workload_measurer_online.py \
  --num-nodes 8 \
  --base-port 8000 \
  --average-interval 0.5 \
  --maxq 50 \
  --dataset-file datasets/mixed.jsonl
```
## Step 4.1 — Dataset suit -  meta-llama/Llama-3.1-8B-Instruct on a6000 and a100
```bash
# sample vllm start
export CUDA_VISIBLE_DEVICES=0 && vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --dtype auto --api-key token-abc123 --enable-prefix-caching --gpu-memory-utilization 0.95

python workload_measurer_online.py \
  --num-nodes 8 \
  --base-port 8000 \
  --average-interval 0.5 \
  --maxq 50 \
  --dataset-file datasets/mixed.jsonl
```
## Step 4.2 — Dataset suit - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B on a100
```bash
# sample vllm start
export CUDA_VISIBLE_DEVICES=0 && vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --port 8000 --dtype auto --api-key token-abc123 --enable-prefix-caching --gpu-memory-utilization 0.95 --max_num_batched_tokens=16384 

python workload_measurer_online.py \
  --num-nodes 2 \
  --base-port 8000 \
  --average-interval 0.5 \
  --maxq 50 \
  --dataset-file datasets/mixed.jsonl
```
## Expected outputs
After sucessful completion, you should see:
```bash
log_<dataset>_<interval>/experiment_results.json
log_<dataset>_<interval>/throughput_records.json
fig_<dataset>_<interval>/*.png
```
 and a printed summary in the terminal and a saved summary text file in the log directory.
 
### Parameter explanation

- `--num-nodes 2`  
  Number of **model servers** (“model nodes”) to use in this run. The script assumes each node is reachable on a distinct port, assigned consecutively starting from `--base-port`.

- `--base-port 8000`  
  Starting port for the first model node. Node ports are assigned as:
  `base-port + i` for `i = 0 .. num-nodes-1`.  
  Example: with `--num-nodes 2` and `--base-port 8000`, the script targets ports **8000** and **8001**.  
  > adjust the IP + port construction in the code accordingly.

- `--average-interval 0.5`  
  Mean inter-arrival time (in seconds) for **Poisson arrivals**. Smaller values generate a higher offered load (more frequent requests).  

- `--maxq 50`  
  Maximum number of requests to issue from the workload file for this epoch.  
  > For each epoch, restart the vLLM servers between runs/epochs to reset cache state and then aggregate the result form all the epoches.

- `--dataset-file datasets/sample.jsonl`  
  Path to the workload dataset in **JSONL** format. Each line must be a JSON object containing a `workload` field, for example:
  ```jsonl
  {"workload": "Explain what TTFT measures."}
  {"workload": "Summarize prefix caching in one paragraph."}
  
 The datasets (aaps.jsonl, toolbech.jsonl, loogle.jsonl, mixed.jsonl) or through [drive](https://drive.google.com/drive/folders/1RDFIjcAnXS59gcg5-bjtwc4uM1f3JSmm):

```bash
pip install - U gdown
gdwon --id 1RDFIjcAnXS59gcg5-bjtwc4uM1f3JSmm
```
 
 In a future release, we plan to package this environment as a container image to make deployment easier and enable more portable, broader experiments across different machines and clusters.

