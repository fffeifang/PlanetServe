import os
import time
import json
import threading
import shutil
from pathlib import Path
import numpy as np
import requests
from transformers import AutoTokenizer
from collections import deque
import random
import matplotlib.pyplot as plt
from openai import OpenAI
import argparse
import math

# Model and tokenization settings
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # model name
TOKENIZER_NAME = MODEL_NAME  # tokenizer
CHUNK_SIZE = 10

# ------------------------------ HashRadix Tree ------------------------------
class HashRadixNode:
    __slots__ = ["children", "is_end", "pass_count", "end_count"]

    def __init__(self) -> None:
        self.children: dict[int, "HashRadixNode"] = {}
        self.is_end: bool = False
        self.pass_count: int = 0
        self.end_count: int = 0

class HashRadix:
    def __init__(self) -> None:
        self.root = HashRadixNode()

    @staticmethod
    def hash_chunk(chunk: tuple[int, ...], mod: int = 1_000_000_007) -> int:
        hash_val = 0
        for tid in chunk:
            hash_val = (hash_val * 31 + tid) % mod
        return hash_val

    def insert(self, chunks: list[tuple[int, ...]]) -> None:
        current = self.root
        current.pass_count += 1
        for chunk in chunks:
            hval = self.hash_chunk(chunk)
            if hval not in current.children:
                current.children[hval] = HashRadixNode()
            current = current.children[hval]
            current.pass_count += 1
        current.is_end = True
        current.end_count += 1

    def delete(self, chunks: list[tuple[int, ...]]) -> bool:
        current = self.root
        stack: list[tuple[HashRadixNode, int, HashRadixNode]] = []
        for chunk in chunks:
            hval = self.hash_chunk(chunk)
            child = current.children.get(hval)
            if child is None:
                return False
            stack.append((current, hval, child))
            current = child

        if current.end_count <= 0:
            return False

        current.end_count -= 1
        if current.end_count == 0:
            current.is_end = False

        for parent, hval, child in reversed(stack):
            child.pass_count -= 1
            if child.pass_count <= 0:
                del parent.children[hval]

        if self.root.pass_count > 0:
            self.root.pass_count -= 1

        return True

# ------------------------------ Model Node -----------------------------------
class ModelNode:
    def __init__(self, name: str, tokenizer: AutoTokenizer, chunk_size: int, build_text: str, max_cached_request: int = 10) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.tree: HashRadix = HashRadix()
        self.chunks: list[tuple[int, ...]] = []
        self.max_cached_request = max_cached_request
        self.workloads = deque(maxlen=self.max_cached_request)
        self.all_workloads: list[str] = []
        self.baseline_chunks: list[tuple[int, ...]] = []
        self.baseline_active: bool = False
        self._build_index_tree(build_text)
        self.port: int = None

    def _text_to_chunks(self, text: str) -> list[tuple[int, ...]]:
        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"].tolist()[0]
        chunks: list[tuple[int, ...]] = []
        for i in range(0, len(input_ids), self.chunk_size):
            chunks.append(tuple(input_ids[i: i + self.chunk_size]))
        return chunks

    def _build_index_tree(self, user_text: str) -> None:
        baseline = self._text_to_chunks(user_text)
        self.chunks.extend(baseline)
        self.tree.insert(baseline)
        self.baseline_chunks = baseline
        self.baseline_active = True

    def add_additional_text(self, additional_text: str) -> None:
        tokens = self.tokenizer(additional_text, return_tensors="pt")
        input_ids = tokens["input_ids"].tolist()[0]
        total_tokens = len(input_ids)
        new_chunks: list[tuple[int, ...]] = []
        i = 0
        while i < total_tokens:
            chunk = tuple(input_ids[i: i + self.chunk_size])
            self.chunks.append(chunk)
            new_chunks.append(chunk)
            i += self.chunk_size
        self.tree.insert(new_chunks)
        print(f"Added {len(new_chunks)} new chunks to the trie.")

    def insert_workload(self, workload: str) -> None:
        if self.baseline_active:
            if self.baseline_chunks:
                self.tree.delete(self.baseline_chunks)
            self.baseline_active = False

        # Evict oldest if cache is full
        if len(self.workloads) == self.max_cached_request:
            evicted_text = self.workloads.popleft()
            evicted_chunks = self._text_to_chunks(evicted_text)
            self.tree.delete(evicted_chunks)

        self.workloads.append(workload)
        self.all_workloads.append(workload)
        new_chunks = self._text_to_chunks(workload)
        self.chunks.extend(new_chunks)
        self.tree.insert(new_chunks)
        print(f"Updated workload cache for {self.name}. Current cache size: {len(self.workloads)}.")

# ------------------------------ Matching & Load Balancing --------------------
def chunk_text(tokenizer: AutoTokenizer, text: str, chunk_size: int) -> list[tuple[int, ...]]:
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].tolist()[0]
    chunks: list[tuple[int, ...]] = []
    for i in range(0, len(input_ids), chunk_size):
        chunks.append(tuple(input_ids[i: i + chunk_size]))
    return chunks

def compute_match_metric_from_chunks(tree: HashRadix, chunks: list[tuple[int, ...]]) -> float:
    current = tree.root
    matched = 0
    for chunk in chunks:
        hval = tree.hash_chunk(chunk)
        if hval in current.children:
            current = current.children[hval]
            matched += 1
        else:
            break
    return matched / float(len(chunks)) if chunks else 0.0

def compute_match_metric(tree: HashRadix, input_text: str, tokenizer: AutoTokenizer, chunk_size: int) -> float:
    return compute_match_metric_from_chunks(tree, chunk_text(tokenizer, input_text, chunk_size))

def find_best_matched_model_node(
    model_nodes: list[ModelNode],
    input_text: str,
    batch_size_tokens: int = None,
    load_table: dict[str, dict[str, float]] = None,
    load_threshold_bf: float = None,
    avg_lat_ms_range: tuple[float, float] = None,
    p99_ms_range: tuple[float, float] = None,
    tokenizer: AutoTokenizer = None
) -> ModelNode:
    # If load balancing is disabled (no load_table), fall back to simple prefix matching
    if load_table is None or batch_size_tokens is None or tokenizer is None:
        node_metric_list = []
        for node in model_nodes:
            metric = compute_match_metric(node.tree, input_text, node.tokenizer, node.chunk_size)
            node_metric_list.append((node, metric))
        metrics = [m for (_, m) in node_metric_list]
        if max(metrics) == min(metrics):
            # If all metrics are identical, choose the node with the fewest workloads
            best_node = min(model_nodes, key=lambda n: len(n.all_workloads))
            best_metric = metrics[0] if metrics else 0.0
            print("All match metrics are identical.")
        else:
            best_node, best_metric = max(node_metric_list, key=lambda x: x[1])
        if best_node is not None:
            print(f"Best matched model: {best_node.name} with metric {best_metric:.3f}")
            best_node.insert_workload(input_text)
        else:
            print("No model node matched the input string.")
        return best_node

    # Load balancing enabled
    batch_size_tokens = max(1, int(batch_size_tokens))
    pre_chunks = chunk_text(tokenizer, input_text, model_nodes[0].chunk_size)
    def batches_for_queue(q: int) -> int:
        return 0 if q <= 0 else (q + batch_size_tokens - 1) // batch_size_tokens

    def node_bf(n: ModelNode) -> float:
        lt = load_table[n.name]
        return lt["recent_avg_latency_ms"] * batches_for_queue(lt["queue_tokens"])

    node_metric_list = [(n, compute_match_metric_from_chunks(n.tree, pre_chunks)) for n in model_nodes]
    bfs = {n.name: node_bf(n) for n in model_nodes}
    hits = [(node, metric) for (node, metric) in node_metric_list if metric > 0.0]

    def p80(values: list[float]) -> float:
        if not values:
            return float("inf")
        s = sorted(values)
        n = len(s)
        k = (n * 80 + 99) // 100  # ceil(0.8 * n)
        idx = max(0, min(n - 1, k - 1))
        return s[idx]

    if not hits:
        # No prefix match: choose node with smallest BF, tiebreak by fewest total workloads
        selected = min(model_nodes, key=lambda n: (bfs[n.name], len(n.all_workloads)))
        decision = f"Cache miss -> LB chose {selected.name} (BF={bfs[selected.name]:.2f})"
    else:
        hit_bfs = [bfs[n.name] for (n, _) in hits]
        threshold = p80(hit_bfs)
        eligible_hits = [(n, m) for (n, m) in hits if bfs[n.name] <= threshold]
        if eligible_hits:
            best_metric = max(m for (_, m) in eligible_hits)
            candidates = [(n, m) for (n, m) in eligible_hits if m == best_metric]
            candidates.sort(key=lambda nm: (bfs[nm[0].name], len(nm[0].all_workloads)))
            selected = candidates[0][0]
            decision = (f"Cache hit -> chose {selected.name} (metric={best_metric:.3f}, "
                        f"BF={bfs[selected.name]:.2f}, dyn_thresh={threshold:.2f})")
        else:
            selected = min(model_nodes, key=lambda n: (bfs[n.name], len(n.all_workloads)))
            decision = (f"Cache hit but all above dyn 80th BF -> LB chose {selected.name} "
                        f"(BF={bfs[selected.name]:.2f}, dyn_thresh={threshold:.2f})")

    # Queuing: add the prompt's tokens to selected node's queue
    tks = len(pre_chunks) * model_nodes[0].chunk_size  # approximate token count
    load_table[selected.name]["queue_tokens"] += int(tks)
    print(decision)

    # Update prefix cache for selected node (prefix tree)
    selected.insert_workload(input_text)
    return selected

# ------------------------------ Workload Reading ----------------------------
def read_workloads_from_jsonl(jsonl_file: str) -> list[str]:
    workloads = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                w = data.get("workload", None)
                if isinstance(w, str):
                    workloads.append(w)
                else:
                    print("Skipping line without valid 'workload' field.")
            except json.JSONDecodeError:
                print("Skipping malformed JSON line.")
    return workloads

# ------------------------------ Metric Monitoring --------------------------
def get_kv_cache_usage(metrics_url: str = None) -> float:
    url = metrics_url or f"http://localhost:{PORT}/metrics"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.startswith("vllm:gpu_cache_usage_perc"):
                    parts = line.split()
                    return float(parts[-1]) if len(parts) >= 2 else 0.0
    except Exception:
        pass
    return 0.0

def get_prefix_cache_hit_rate(metrics_url: str = None) -> float:
    url = metrics_url or f"http://localhost:{PORT}/metrics"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            total_q = total_h = None
            for line in resp.text.splitlines():
                if line.startswith("vllm:gpu_prefix_cache_queries_total"):
                    total_q = float(line.split()[-1])
                elif line.startswith("vllm:gpu_prefix_cache_hits_total"):
                    total_h = float(line.split()[-1])
            if total_q and total_q > 0 and total_h is not None:
                return total_h / total_q
    except Exception:
        pass
    return 0.0

def get_time_per_output_token(metrics_url: str = None) -> float:
    url = metrics_url or f"http://localhost:{PORT}/metrics"
    tpot_sum = None
    tpot_count = None
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if line.startswith("vllm:time_per_output_token_seconds_sum"):
                tpot_sum = float(line.split()[-1])
            elif line.startswith("vllm:time_per_output_token_seconds_count"):
                tpot_count = float(line.split()[-1])
        if tpot_sum is not None and tpot_count and tpot_count > 0:
            return tpot_sum / tpot_count
    except Exception:
        pass
    return 0.0

def get_time_to_first_token(metrics_url: str = None) -> float:
    url = metrics_url or f"http://localhost:{PORT}/metrics"
    ttft_sum = None
    ttft_count = None
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            if line.startswith("vllm:time_to_first_token_seconds_sum"):
                ttft_sum = float(line.split()[-1])
            elif line.startswith("vllm:time_to_first_token_seconds_count"):
                ttft_count = float(line.split()[-1])
        if ttft_sum is not None and ttft_count and ttft_count > 0:
            return ttft_sum / ttft_count
    except Exception:
        pass
    return 0.0

def get_cumulative_counters_for_port(port: int) -> tuple[float, float, float]:
    prompt_tokens = gen_tokens = req_success = 0.0
    url = f"http://localhost:{port}/metrics"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.startswith("vllm:prompt_tokens_total"):
                    prompt_tokens += float(line.split()[-1])
                elif line.startswith("vllm:generation_tokens_total"):
                    gen_tokens += float(line.split()[-1])
                elif line.startswith("vllm:request_success_total"):
                    req_success += float(line.split()[-1])
    except Exception:
        pass
    return prompt_tokens, gen_tokens, req_success

def monitor_metric(metric_func: callable, records: list[tuple[float, float]], stop_evt: threading.Event, start_time: float, interval: float = 0.1):
    while not stop_evt.is_set():
        elapsed = time.time() - start_time
        records.append((elapsed, metric_func()))
        time.sleep(interval)

def monitor_throughput(port_list: list[int], records: list[tuple[float, float, float, float]], stop_evt: threading.Event, start_time: float, interval: float = 0.2):
    while not stop_evt.is_set():
        elapsed = time.time() - start_time
        total_p = total_g = total_r = 0.0
        for port in port_list:
            p, g, r = get_cumulative_counters_for_port(port)
            total_p += p; total_g += g; total_r += r
        records.append((elapsed, total_p, total_g, total_r))
        time.sleep(interval)

def query_model(prompt: str, port: int, max_new_tokens: int) -> tuple[str, float]:
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="token-abc123")
    start = time.time()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens
    )
    return getattr(completion, "id", "unknown"), time.time() - start

# Run a single query on a given model server (with metrics monitoring)
def run_single_query(prompt: str, port: int, max_new_tokens: int) -> dict:
    usage_records: list[tuple[float, float]] = []
    prefix_records: list[tuple[float, float]] = []
    tpot_records: list[tuple[float, float]] = []
    ttft_records: list[tuple[float, float]] = []
    stop_evt = threading.Event()
    start_time = time.time()
    # Start metric monitoring threads for this query
    threads = []
    usage_thread = threading.Thread(target=monitor_metric, args=(lambda: get_kv_cache_usage(f"http://localhost:{port}/metrics"), usage_records, stop_evt, start_time))
    prefix_thread = threading.Thread(target=monitor_metric, args=(lambda: get_prefix_cache_hit_rate(f"http://localhost:{port}/metrics"), prefix_records, stop_evt, start_time))
    tpot_thread = threading.Thread(target=monitor_metric, args=(lambda: get_time_per_output_token(f"http://localhost:{port}/metrics"), tpot_records, stop_evt, start_time))
    ttft_thread = threading.Thread(target=monitor_metric, args=(lambda: get_time_to_first_token(f"http://localhost:{port}/metrics"), ttft_records, stop_evt, start_time))
    threads.extend([usage_thread, prefix_thread, tpot_thread, ttft_thread])
    for t in threads:
        t.daemon = True
        t.start()
    # Query the model server
    qid, gen_time = query_model(prompt, port, max_new_tokens)
    # Stop metric monitoring threads
    stop_evt.set()
    for t in threads:
        t.join()
    # Compile results for this query
    return {
        "query_id": qid,
        "generation_time": gen_time,
        "final_kv_cache_usage": usage_records[-1][1] if usage_records else 0.0,
        "final_prefix_cache_hit_rate": prefix_records[-1][1] if prefix_records else 0.0,
        "final_time_per_output_token": tpot_records[-1][1] if tpot_records else 0.0,
        "final_time_to_first_token": ttft_records[-1][1] if ttft_records else 0.0,
        "usage_records": usage_records,
        "prefix_usage_records": prefix_records,
        "time_per_output_token_records": tpot_records,
        "time_to_first_token_records": ttft_records,
        "absolute_start": start_time
    }

# Schedule and run queries (Poisson arrivals) with load-balanced dispatching
def schedule_and_run(
    prompts: list[str],
    average_interval: float,
    model_nodes: list[ModelNode],
    batch_size_tokens: int,
    avg_lat_ms_range: tuple[float, float],
    p99_ms_range: tuple[float, float],
    load_table: dict,
    service_tokens_per_step: int,
    dispatch_strategy: str,
    max_new_tokens: int
) -> tuple[list[dict], list[tuple[float, float, float, float]]]:
    n = len(prompts)
    if n == 0:
        return [], []
    # Generate Poisson arrival schedule
    arrivals = np.cumsum(np.random.exponential(average_interval, size=n))
    results: list[dict] = []
    results_lock = threading.Lock()
    # Start throughput monitoring thread
    port_list = [node.port for node in model_nodes]
    throughput_records: list[tuple[float, float, float, float]] = []
    th_stop_evt = threading.Event()
    experiment_start = time.time()
    th_thread = threading.Thread(target=monitor_throughput, args=(port_list, throughput_records, th_stop_evt, experiment_start))
    th_thread.daemon = True
    th_thread.start()
    # Scheduler loop: dispatch each query at its scheduled time
    sched_lock = threading.Lock()
    query_threads: list[threading.Thread] = []
    for idx, prompt in enumerate(prompts):
        target_time = arrivals[idx]
        delay = target_time - (time.time() - experiment_start)
        if delay > 0:
            time.sleep(delay)
        print(f"[Query {idx+1}] dispatching at {time.time()-experiment_start:.2f}s")

        if dispatch_strategy == "lb":
            with sched_lock:
                selected_node = find_best_matched_model_node(
                    model_nodes,
                    prompt,
                    batch_size_tokens=batch_size_tokens,
                    load_table=load_table,
                    load_threshold_bf=float("inf"),
                    avg_lat_ms_range=avg_lat_ms_range,
                    p99_ms_range=p99_ms_range,
                    tokenizer=model_nodes[0].tokenizer
                )
                # Simulate processing: drain a batch of tokens from each node's queue
                for node in model_nodes:
                    lt = load_table[node.name]
                    lt["queue_tokens"] = max(0, lt["queue_tokens"] - service_tokens_per_step)
        else:
            # Round-robin: bypass scheduler/LB entirely
            selected_node = model_nodes[idx % len(model_nodes)]
            selected_node.insert_workload(prompt)

        # Launch a thread to execute the query on the chosen node
        def worker(prompt=prompt, node=selected_node, qnum=idx+1):
            res = run_single_query(prompt, node.port, max_new_tokens)
            res["query_number"] = qnum
            res["node_name"] = node.name
            with results_lock:
                results.append(res)
                # Update latency metrics for this node using actual observed latency
                lt = load_table[node.name]
                actual_ms = res["generation_time"] * 1000.0  # convert seconds to ms
                lt["most_recent_latency_ms"] = actual_ms
                # Update exponential moving average for latency
                if lt.get("recent_avg_latency_ms", 0) <= 0:
                    lt["recent_avg_latency_ms"] = actual_ms
                else:
                    lt["recent_avg_latency_ms"] = lt["recent_avg_latency_ms"] * 0.875 + actual_ms * 0.125
                if "latencies" not in lt:
                    lt["latencies"] = []
                lt["latencies"].append(actual_ms)
                if len(lt["latencies"]) > 50:  # cap history to latest 50 samples
                    lt["latencies"].pop(0)
                sorted_hist = sorted(lt["latencies"])
                idx99 = math.ceil(0.99 * len(sorted_hist)) - 1
                if idx99 < 0:
                    idx99 = 0
                if idx99 >= len(sorted_hist):
                    idx99 = len(sorted_hist) - 1
                lt["p99_latency_ms"] = sorted_hist[idx99]
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        query_threads.append(t)
    # Wait for all query threads to finish
    for t in query_threads:
        t.join()
    # Stop throughput monitoring thread
    th_stop_evt.set()
    th_thread.join()
    return results, throughput_records

# Plotting and summary functions
def plot_latency_stats_from_log(log_file: str, output_dir: str, experiment_name: str) -> None:
    with open(log_file, 'r', encoding='utf-8') as f:
        qr = json.load(f)
    if not qr:
        return
    qnums = [r['query_number'] for r in qr]
    avgs = [np.mean([r2['generation_time'] for r2 in qr[:i+1]]) for i in range(len(qr))]
    p99s = [np.percentile([r2['generation_time'] for r2 in qr[:i+1]], 99) for i in range(len(qr))]
    plt.figure()
    plt.plot(qnums, avgs, 'o-', label='Avg')
    plt.plot(qnums, p99s, 'o-', label='P99')
    plt.xlabel('Query Number'); plt.ylabel('Latency (s)'); plt.title(f"{experiment_name} Latency"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'latency_stats.png')); plt.close()

def plot_prefix_hit_rate_from_log(log_file: str, output_dir: str, experiment_name: str) -> None:
    with open(log_file, 'r', encoding='utf-8') as f:
        qr = json.load(f)
    if not qr:
        return
    qnums, rates = [], []
    for r in qr:
        qnums.append(r['query_number'])
        recs = r.get('prefix_usage_records', [])
        rates.append(np.mean([v for (_, v) in recs]) if recs else 0)
    plt.figure()
    plt.plot(qnums, rates, 'o-')
    plt.xlabel('Query Number'); plt.ylabel('Prefix Hit Rate'); plt.title(f"{experiment_name} Prefix Hit Rate"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prefix_hit_rate.png')); plt.close()

def plot_arrival_distribution_from_log(log_file: str, output_dir: str, experiment_name: str) -> None:
    with open(log_file, 'r', encoding='utf-8') as f:
        qr = json.load(f)
    times = sorted(r['absolute_start'] for r in qr)
    if len(times) < 2:
        return
    intervals = np.diff(times)
    plt.figure()
    plt.hist(intervals, bins=15, edgecolor='black')
    plt.xlabel('Inter-arrival Time (s)'); plt.ylabel('Frequency'); plt.title(f"{experiment_name} Arrivals"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'arrival_distribution.png')); plt.close()

def plot_token_throughput_from_log(throughput_log: str, output_dir: str, experiment_name: str) -> None:
    with open(throughput_log, 'r', encoding='utf-8') as f:
        arr = json.load(f)
    recs = [(e['elapsed'], e['prompt_tokens'], e['gen_tokens'], e['request_success']) for e in arr]
    if len(recs) < 2:
        return
    times, rates = [], []
    for i in range(1, len(recs)):
        t0, p0, g0, _ = recs[i-1]; t1, p1, g1, _ = recs[i]
        dt = t1 - t0
        if dt > 0:
            times.append(t1)
            rates.append(((p1 + g1) - (p0 + g0)) / dt)
    avg = np.mean(rates) if rates else 0.0
    plt.figure()
    plt.plot(times, rates, 'o-')
    plt.axhline(avg, linestyle='--', label=f'Avg {avg:.2f}')
    plt.xlabel('Time (s)'); plt.ylabel('Tokens/s'); plt.title(f"{experiment_name} Token Throughput"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'token_throughput.png')); plt.close()

def plot_request_throughput_from_log(throughput_log: str, output_dir: str, experiment_name: str) -> None:
    with open(throughput_log, 'r', encoding='utf-8') as f:
        arr = json.load(f)
    recs = [(e['elapsed'], e['prompt_tokens'], e['gen_tokens'], e['request_success']) for e in arr]
    if len(recs) < 2:
        return
    times, rates = [], []
    for i in range(1, len(recs)):
        t0, _, _, r0 = recs[i-1]; t1, _, _, r1 = recs[i]
        dt = t1 - t0
        if dt > 0:
            times.append(t1)
            rates.append((r1 - r0) / dt)
    avg = np.mean(rates) if rates else 0.0
    plt.figure()
    plt.plot(times, rates, 'o-')
    plt.axhline(avg, linestyle='--', label=f'Avg {avg:.2f}')
    plt.xlabel('Time (s)'); plt.ylabel('Req/s'); plt.title(f"{experiment_name} Request Throughput"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'request_throughput.png')); plt.close()

def plot_tpot_stats_from_log(log_file: str, output_dir: str, experiment_name: str) -> None:
    with open(log_file, 'r', encoding='utf-8') as f:
        qr = json.load(f)
    if not qr:
        return
    qnums, vals = [], []
    for r in qr:
        qnums.append(r['query_number'])
        recs = r.get('time_per_output_token_records', [])
        vals.append(np.mean([v for (_, v) in recs]) if recs else 0)
    plt.figure()
    plt.plot(qnums, vals, 'o-')
    plt.xlabel('Query Number'); plt.ylabel('TPOT (s)'); plt.title(f"{experiment_name} Time Per Output Token"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'tpot_stats.png')); plt.close()

def plot_ttft_stats_from_log(log_file: str, output_dir: str, experiment_name: str) -> None:
    with open(log_file, 'r', encoding='utf-8') as f:
        qr = json.load(f)
    if not qr:
        return
    qnums, vals = [], []
    for r in qr:
        qnums.append(r['query_number'])
        recs = r.get('time_to_first_token_records', [])
        vals.append(np.mean([v for (_, v) in recs]) if recs else 0)
    plt.figure()
    plt.plot(qnums, vals, 'o-')
    plt.xlabel('Query Number'); plt.ylabel('TTFT (s)'); plt.title(f"{experiment_name} Time To First Token"); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ttft_stats.png')); plt.close()

def summarize_results(log_dir: str, experiment_name: str, use_poisson: bool, average_interval: float) -> None:
    exp_file = os.path.join(log_dir, 'experiment_results.json')
    thr_file = os.path.join(log_dir, 'throughput_records.json')
    with open(exp_file, 'r', encoding='utf-8') as f:
        exp_data = json.load(f)
    lat = [r['generation_time'] for r in exp_data]
    avg_lat = np.mean(lat) if lat else 0.0
    p99_lat = np.percentile(lat, 99) if lat else 0.0
    tpot_vals = [np.mean([np.mean([v for (_, v) in r.get('time_per_output_token_records', [])]) for r in exp_data if r.get('time_per_output_token_records')])]
    ttft_vals = [np.mean([np.mean([v for (_, v) in r.get('time_to_first_token_records', [])]) for r in exp_data if r.get('time_to_first_token_records')])]
    avg_tpot = tpot_vals[0] if tpot_vals and tpot_vals[0] else 0.0
    avg_ttft = ttft_vals[0] if ttft_vals and ttft_vals[0] else 0.0
    hit_rates = [np.mean([v for (_, v) in r.get('prefix_usage_records', [])]) for r in exp_data if r.get('prefix_usage_records')]
    avg_hit = np.mean(hit_rates) if hit_rates else 0.0
    arrivals = sorted(r['absolute_start'] for r in exp_data)
    inter_arrivals = [arrivals[i] - arrivals[i-1] for i in range(1, len(arrivals))]
    avg_ia = np.mean(inter_arrivals) if inter_arrivals else 0.0
    with open(thr_file, 'r', encoding='utf-8') as f:
        thr_data = json.load(f)
    if len(thr_data) >= 2:
        t0 = thr_data[0]['elapsed']; p0 = thr_data[0]['prompt_tokens']; g0 = thr_data[0]['gen_tokens']; r0 = thr_data[0]['request_success']
        t1 = thr_data[-1]['elapsed']; p1 = thr_data[-1]['prompt_tokens']; g1 = thr_data[-1]['gen_tokens']; r1 = thr_data[-1]['request_success']
        tok_rate = ((p1 + g1) - (p0 + g0)) / (t1 - t0) if (t1 - t0) > 0 else 0.0
        req_rate = (r1 - r0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
    else:
        tok_rate = req_rate = 0.0
    lines = [
        "===== Experiment Summary =====",
        f"Name: {experiment_name}",
        f"Poisson: {use_poisson}",
        f"Mean Interval: {average_interval:.3f}s",
        f"Avg Latency: {avg_lat:.3f}s",
        f"P99 Latency: {p99_lat:.3f}s",
        f"TPOT: {avg_tpot:.3f}s",
        f"TTFT: {avg_ttft:.3f}s",
        f"Avg Token Throughput: {tok_rate:.2f} tok/s",
        f"Avg Request Throughput: {req_rate:.2f} req/s",
        f"Avg Prefix Hit Rate: {avg_hit:.3f}",
        f"Avg Inter-arrival: {avg_ia:.3f}s",
        "=============================="
    ]
    print("\n" + "\n".join(lines) + "\n")
    with open(os.path.join(log_dir, f"{experiment_name}_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Online vLLM workload scheduler and monitor")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of model servers (GPUs) to use (default: 1)")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port for model servers (default: 8000). Additional servers on consecutive ports.")
    parser.add_argument("--average-interval", type=float, default=0.16, help="Mean inter-arrival time for Poisson scheduling in seconds (default: 0.16)")
    parser.add_argument("--maxq", type=int, default=50, help="Maximum number of queries to send (default: 50)")
    parser.add_argument("--dataset-file", type=str, default=None, help="Path to a JSONL file containing workloads (prompts)")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max generated tokens per request (default: 50)")
    parser.add_argument("--dispatch", choices=["lb", "rr"], default="lb",
                        help="Dispatch policy: 'lb' (scheduler+load-balancer) or 'rr' (roundrobin bypass LB)")
    args = parser.parse_args()

    num_nodes = args.num_nodes
    global PORT
    PORT = args.base_port
    base_port = args.base_port
    avg_interval = args.average_interval
    max_queries = args.maxq
    dataset_file = args.dataset_file
    max_new_tokens = args.max_new_tokens
    dispatch_strategy = args.dispatch

    # Prepare prompt list
    prompts: list[str] = []
    if dataset_file:
        if not os.path.exists(dataset_file):
            print(f"Dataset file not found: {dataset_file}")
            return
        prompts = read_workloads_from_jsonl(dataset_file)
    else:
        print("No dataset specified. Use --dataset-file")
        return

    if max_queries is not None and max_queries > 0:
        prompts = prompts[:max_queries]
    if not prompts:
        print("No workloads found for querying.")
        return

    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    build_text = "I am a model node."
    model_nodes = [ModelNode(f"mnode{i+1}", tokenizer, CHUNK_SIZE, build_text) for i in range(num_nodes)]
    # Assign ports to model nodes
    for i, node in enumerate(model_nodes):
        node.port = base_port + i

    # Initialize load balancing table
    avg_latency = 71.0
    p99_latency = 78.0
    factor = p99_latency / avg_latency if avg_latency > 0 else 1.0
    lo = avg_latency / factor
    hi = avg_latency * factor
    AVG_LAT_MS_RANGE = (min(lo, hi), max(lo, hi))
    P99_LAT_MS_RANGE = (
        float(os.getenv("P99_LAT_MS_MIN", "80.0")),
        float(os.getenv("P99_LAT_MS_MAX", "90.0")),
    )
    LOAD_THRESHOLD_BF = float(os.getenv("LOAD_THRESHOLD_BF", "inf"))
    BATCH_SIZE_TOKENS = int(os.getenv("BATCH_SIZE_TOKENS", "2048"))
    SERVICE_TOKENS_PER_STEP = int(os.getenv("SERVICE_TOKENS_PER_STEP", str(BATCH_SIZE_TOKENS)))

    load_table = {
        node.name: {
            "queue_tokens": 0,
            "most_recent_latency_ms": random.uniform(AVG_LAT_MS_RANGE[0], AVG_LAT_MS_RANGE[1]),
            "recent_avg_latency_ms": 0.0,
            "p99_latency_ms": random.uniform(P99_LAT_MS_RANGE[0], P99_LAT_MS_RANGE[1])
        }
        for node in model_nodes
    }
    for node in model_nodes:
        lt = load_table[node.name]
        lt["recent_avg_latency_ms"] = lt["most_recent_latency_ms"]

    # Create output directories for logs and plots
    exp_name = Path(dataset_file).stem
    log_dir = f"log_{exp_name}_{avg_interval}"
    fig_dir = f"fig_{exp_name}_{avg_interval}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    # Run experiment
    print("=== Running online scheduled experiment ===")
    results, throughput_records = schedule_and_run(
        prompts, avg_interval, model_nodes,
        batch_size_tokens=BATCH_SIZE_TOKENS,
        avg_lat_ms_range=AVG_LAT_MS_RANGE,
        p99_ms_range=P99_LAT_MS_RANGE,
        load_table=load_table,
        service_tokens_per_step=SERVICE_TOKENS_PER_STEP,
        dispatch_strategy=dispatch_strategy,
        max_new_tokens=max_new_tokens
    )
    results.sort(key=lambda x: x.get('query_number', 0))
    with open(os.path.join(log_dir, 'experiment_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(log_dir, 'throughput_records.json'), 'w', encoding='utf-8') as f:
        json.dump([{'elapsed': e, 'prompt_tokens': p, 'gen_tokens': g, 'request_success': r} for (e, p, g, r) in throughput_records], f, ensure_ascii=False, indent=2)

    # Generate plots and summary
    plot_latency_stats_from_log(os.path.join(log_dir, 'experiment_results.json'), fig_dir, exp_name)
    plot_prefix_hit_rate_from_log(os.path.join(log_dir, 'experiment_results.json'), fig_dir, exp_name)
    plot_arrival_distribution_from_log(os.path.join(log_dir, 'experiment_results.json'), fig_dir, exp_name)
    plot_token_throughput_from_log(os.path.join(log_dir, 'throughput_records.json'), fig_dir, exp_name)
    plot_request_throughput_from_log(os.path.join(log_dir, 'throughput_records.json'), fig_dir, exp_name)
    plot_tpot_stats_from_log(os.path.join(log_dir, 'experiment_results.json'), fig_dir, exp_name)
    plot_ttft_stats_from_log(os.path.join(log_dir, 'experiment_results.json'), fig_dir, exp_name)
    summarize_results(log_dir, exp_name, True, avg_interval)

if __name__ == "__main__":
    main()
