import aiohttp
import asyncio
import json
from time import perf_counter
import numpy as np
from transformers import AutoTokenizer
import multiprocessing
from HashRadixTree.HashRadixTree import HashRadixTree
from HashRadixTree.ModelNode import ModelNode
import argparse
# ----------- Configuration -----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="../../models/meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-model-concurrency", type=int, default=4)
    p.add_argument("--sched-workers", type=int, default=1)
    p.add_argument("--serve-workers", type=int, default=256)
    p.add_argument("--request-rate", type=float, default=72.0)
    p.add_argument("--dataset-path", type=str, default="../../datasets/toolbench_zipf_1.1_prompts_6000.jsonl")
    return p.parse_args()

args = parse_args()

MODEL_NAME = args.model_name
MAX_MODEL_CONCURRENCY = args.max_model_concurrency
SCHED_WORKERS = args.sched_workers
SERVE_WORKERS = args.serve_workers
REQUEST_RATE = args.request_rate
DATASET_PATH = args.dataset_path


with open(DATASET_PATH, "r") as f:
    raw_samples = [json.loads(line) for line in f]
magic_ids = []

build_text = ""

# shared tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=""
)

# ----------- Shared Objects -----------
model_list = [
    ModelNode(f"mnode{i}", f"http://127.0.0.1:800{i}/v1/completions", max_concurrency=MAX_MODEL_CONCURRENCY)
    for i in range(8)
]
hrt = None

print("=====hrt initialized=====")
hrt_ready = asyncio.Lock()

Q1_MS, Q2_MS, INF_MS, E2E_MS = [], [], [], []
# sched_q = asyncio.Queue(maxsize=SERVE_WORKERS)
sched_q = asyncio.Queue(maxsize=8)
serve_q = asyncio.Queue(maxsize=SERVE_WORKERS)


async def monitor_queues():
    while True:
        await asyncio.sleep(30) 
        print(f"[Monitor] sched_q: {sched_q.qsize()}, serve_q: {serve_q.qsize()}")

async def sched_worker():
    global hrt
    while True:
        try:
            idx, sample, t0 = await sched_q.get()
            t1 = perf_counter()
            Q1_MS.append((t1 - t0) * 1000)

            ### add load balancing
            txt = sample['text'][len(build_text):]
            # print(txt[len(build_text):len(build_text)+20])
            ids = tokenizer(txt, return_tensors="pt")["input_ids"][0].tolist()
            tokens = ids
            # lazily initialize HRT on first task
            if hrt is None:
                async with hrt_ready:
                    if hrt is None:
                        hrt = HashRadixTree(
                            first_workload=tokens,
                            first_mnode=model_list[0],
                            candidate_models=model_list
                        )
                        match_model = model_list[0]
            else:
                # print(tokens[:20])
                match_model, hrt_node = hrt.find_match_model(tokens)
                match_factor = match_model.latency_factor()
                pairs = [(m, m.latency_factor()) for m in model_list if m.latency_factor() != 0]
                n = len(pairs)

                if n > 1:
                    pairs = sorted(pairs, key=lambda kv: kv[1])
                    cutoff_idx = int(0.8 * n)

                    if match_factor > pairs[cutoff_idx][1]:
                        match_model=hrt.load_balance(hrt_node, pairs)

            t2_sched = perf_counter()
        
            await serve_q.put((idx, sample, t0, t1, t2_sched, match_model))
        except Exception as e:
            print(f"[sched_worker] idx={idx} error: {e!r}")
        finally:
            sched_q.task_done()

async def serve_worker(session: aiohttp.ClientSession):
    while True:
        try:
            idx, sample, t0, t1, t2_sched, model_node = await serve_q.get()
            t3 = perf_counter()
            Q2_MS.append((t3 - t2_sched) * 1000)

            await model_node.add_task()
            inf_start = perf_counter()
            async with session.post(model_node.url, json={
                "model": model_node.name,
                "prompt": sample['text'],
                "max_tokens": 128,
                "temperature": sample.get('sampling_params', {}).get('temperature', 0),
                "stop": None,
                "echo": False
            }, timeout=80) as resp:
                _ = await resp.json(content_type=None)
            INF_MS.append((perf_counter() - inf_start) * 1000)
            await model_node.finish_task()
        except Exception as e:
            print(f"[sched_worker] idx={idx} error: {e!r}")
        finally:
            serve_q.task_done()
            t4 = perf_counter()
            E2E_MS.append((t4 - t1) * 1000)
            model_node.push_latency(E2E_MS[-1])
            pairs = [(m.name, m.latency_factor()) for m in model_list if m.latency_factor() != 0]
            # print(pairs)
            print(f"model:{model_node.name} is processing idx:{idx} E2E: {E2E_MS[-1]:.1f}ms")

async def main():
    print("=====main started=====")
    async with aiohttp.ClientSession() as session:
        scheders = [asyncio.create_task(sched_worker()) for _ in range(SCHED_WORKERS)]
        servers = [asyncio.create_task(serve_worker(session)) for _ in range(SERVE_WORKERS)]
        print("=====scheders and servers started=====")
        for idx, sample in enumerate(raw_samples):
            await asyncio.sleep(np.random.exponential(1/REQUEST_RATE))
            t0 = perf_counter()
            await sched_q.put((idx, sample, t0))

        # await sched_q.join()
        print("sched unfinished =", getattr(sched_q, "_unfinished_tasks", None))
        await serve_q.join()
        print("serve unfinished =", getattr(serve_q, "_unfinished_tasks", None))
        
        print("----- Perf Summary -----")
        print(f"P50 Q1: {np.percentile(Q1_MS,50):.1f}ms, Q2: {np.percentile(Q2_MS,50):.1f}ms")
        print(f"P50 Inf: {np.percentile(INF_MS,50):.1f}ms, E2E: {np.percentile(E2E_MS,50):.1f}ms")
        print(f"P99 E2E: {np.percentile(E2E_MS,99):.1f}ms, Mean E2E: {np.mean(E2E_MS):.1f}ms")

        for t in scheders + servers:
            t.cancel()
        await asyncio.gather(*scheders, *servers, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
