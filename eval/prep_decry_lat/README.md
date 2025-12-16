## Latency of Preparation and Decryption 
This experiment measures the latency of clove preparation on model nodes and clove decryption on user nodes.
### Setup and Execution
Note: Only model nodes are required to download the LLM weights. User nodes do not need the model files.
Model Download (Model Nodes Only)
```bash
hf download bartowski/Llama-3.3-70B-Instruct-GGUF \
  Llama-3.3-70B-Instruct-Q4_0.gguf \
  --local-dir ../../models \
```
Dataset Download (All Nodes)
```bash
cd datasets
gdown --id 1uhCy0iCjejuZ2TQXR4kq7isSH7U9OLkz
```
Run Evaluation
```bash
sudo chmod +x run_and_visualize.sh
sudo chmod +x run_standalone_eval.sh
./run_and_visualize.sh
```
###  Latency of preparing cloves on model nodes 
Clove preparation latency is measured on model node equipped with a single NVIDIA A100 40GB SXM4 GPU, 30 vCPUs, 200 GiB RAM, and 0.5 TiB SSD running Llama-3.3-70B.

###  decrypting cloves on user nodes.
Clove decryption latency is measured on user nodes equipped with Intel Core i7-7700 @ 3.60 GHz
