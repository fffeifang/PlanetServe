#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


# -----------------------------
# Core simulation
# -----------------------------
def simulate_confidentiality_mc(
    f: float,
    n: int,
    k: int,
    L: int,
    runs: int,
    N: int,
    use_snp: bool,
    rng: random.Random,
) -> float:
    """
    Monte Carlo confidentiality under the simplified capture model.

    Model:
      - Each clove corresponds to one path; path is 'captured' with
            p_capture = 1 - (1-f)^L.
      - If captured, it is observed by exactly one malicious node, uniformly
        from M = round(f*N) malicious nodes.

    Attack success:
      - No SNP: success iff total captured cloves >= k (colluding adversary).
      - SNP: success iff any single malicious node observes >= k cloves.

    Returns:
      confidentiality = P[attack fails] estimated by Monte Carlo.
    """
    M = max(1, int(round(f * N)))
    p_capture = 1.0 - (1.0 - f) ** L

    fail = 0
    for _ in range(runs):
        # record which malicious node saw each captured clove (or None)
        captured_by = []
        for _ in range(n):
            if rng.random() < p_capture:
                captured_by.append(rng.randint(1, M))
            else:
                captured_by.append(None)

        if not use_snp:
            total_captured = sum(x is not None for x in captured_by)
            if total_captured < k:
                fail += 1
        else:
            freq: Dict[int, int] = {}
            for m_id in captured_by:
                if m_id is not None:
                    freq[m_id] = freq.get(m_id, 0) + 1
            success = any(cnt >= k for cnt in freq.values())
            if not success:
                fail += 1

    return fail / runs


def smart_format(x, _):
    return f"{x:.3g}"



def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[float]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.10g}" for v in r) + "\n")


@dataclass
class Params:
    N: int
    runs: int
    seed: int
    f_min: float
    f_max: float
    f_points: int
    n_gc: int
    k_gc: int
    L_gc: int
    n_ps: int
    k_ps: int
    L_ps: int
    output_dir: str
    show: bool


# -----------------------------
# Plot
# -----------------------------
def plot_confidentiality(
    f_vals: np.ndarray,
    series: Dict[str, List[float]],
    out_pdf: str,
    show: bool,
):
    colors = ['#15498D', '#E63660']  # PS, GC
    fontsize_axes = 10
    fontsize_ticks = 8
    fontsize_legend = 8
    linewidth = 1
    markeredgewidth = 1.02

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.set_position([0.15, 0.25, 0.8, 0.68])

    # PlanetServe
    ax.plot(
        f_vals, series["PlanetServe (SNP)"],
        marker='o', color=colors[0], linewidth=linewidth,
        markeredgewidth=markeredgewidth, markerfacecolor='none',
        linestyle='-', clip_on=False, label="PlanetServe"
    )
    ax.plot(
        f_vals, series["PlanetServe (no SNP)"],
        marker='+', color=colors[0], linewidth=linewidth,
        markeredgewidth=markeredgewidth, markerfacecolor='none',
        linestyle='-', clip_on=False, label="PlanetServe BFD"
    )

    # Garlic Cast
    ax.plot(
        f_vals, series["Garlic Cast (SNP)"],
        marker='o', color=colors[1], linewidth=linewidth,
        markeredgewidth=markeredgewidth, markerfacecolor='none',
        linestyle=':', clip_on=False, label="Garlic Cast"
    )
    ax.plot(
        f_vals, series["Garlic Cast (no SNP)"],
        marker='+', color=colors[1], linewidth=linewidth,
        markeredgewidth=markeredgewidth, markerfacecolor='none',
        linestyle=':', clip_on=False, label="Garlic Cast BFD"
    )

    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.1])
    ax.xaxis.set_major_formatter(FuncFormatter(smart_format))

    ax.set_xlabel("Fraction of malicious nodes (f)", fontsize=fontsize_axes)
    ax.set_ylabel("Confidentiality", fontsize=fontsize_axes)

    ax.tick_params(axis='x', labelsize=fontsize_ticks, bottom=False)
    ax.tick_params(axis='y', labelsize=fontsize_ticks)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    ax.set_xlim(f_vals[0], f_vals[-1])

    ax.legend(
        loc='lower left',
        bbox_to_anchor=(0.1, 0.15),
        fontsize=fontsize_legend,
        frameon=True,
        edgecolor='black',
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Artifact-grade confidentiality simulator (Monte Carlo).")
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--runs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--f-min", type=float, default=0.001)
    ap.add_argument("--f-max", type=float, default=0.1)
    ap.add_argument("--f-points", type=int, default=10)

    # Garlic Cast params
    ap.add_argument("--gc-n", type=int, default=4)
    ap.add_argument("--gc-k", type=int, default=3)
    ap.add_argument("--gc-L", type=int, default=6)

    # PlanetServe params
    ap.add_argument("--ps-n", type=int, default=4)
    ap.add_argument("--ps-k", type=int, default=3)
    ap.add_argument("--ps-L", type=int, default=4)

    ap.add_argument("--output-dir", type=str, default="out")
    ap.add_argument("--show", action="store_true")

    args = ap.parse_args()

    params = Params(
        N=args.N,
        runs=args.runs,
        seed=args.seed,
        f_min=args.f_min,
        f_max=args.f_max,
        f_points=args.f_points,
        n_gc=args.gc_n,
        k_gc=args.gc_k,
        L_gc=args.gc_L,
        n_ps=args.ps_n,
        k_ps=args.ps_k,
        L_ps=args.ps_L,
        output_dir=args.output_dir,
        show=args.show,
    )

    ensure_dir(params.output_dir)

    # deterministic RNG for artifact reproducibility
    rng = random.Random(params.seed)

    f_vals = np.linspace(params.f_min, params.f_max, params.f_points)

    # run simulations
    series = {
        "Garlic Cast (no SNP)": [],
        "Garlic Cast (SNP)": [],
        "PlanetServe (no SNP)": [],
        "PlanetServe (SNP)": [],
    }

    for f in f_vals:
        # GC
        series["Garlic Cast (no SNP)"].append(
            simulate_confidentiality_mc(f, params.n_gc, params.k_gc, params.L_gc, params.runs, params.N, False, rng)
        )
        series["Garlic Cast (SNP)"].append(
            simulate_confidentiality_mc(f, params.n_gc, params.k_gc, params.L_gc, params.runs, params.N, True, rng)
        )
        # PS
        series["PlanetServe (no SNP)"].append(
            simulate_confidentiality_mc(f, params.n_ps, params.k_ps, params.L_ps, params.runs, params.N, False, rng)
        )
        series["PlanetServe (SNP)"].append(
            simulate_confidentiality_mc(f, params.n_ps, params.k_ps, params.L_ps, params.runs, params.N, True, rng)
        )

    # export CSV
    out_csv = os.path.join(params.output_dir, "confidentiality.csv")
    header = ["f"] + list(series.keys())
    rows = []
    for i, f in enumerate(f_vals):
        rows.append([float(f)] + [float(series[k][i]) for k in series.keys()])
    write_csv(out_csv, header, rows)

    # export params
    out_params = os.path.join(params.output_dir, "params.json")
    meta = {
        "params": asdict(params),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": {
            "csv": "confidentiality.csv",
            "pdf": "confidentiality.pdf",
        },
    }
    with open(out_params, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # plot
    out_pdf = os.path.join(params.output_dir, "confidentiality.pdf")
    plot_confidentiality(f_vals, series, out_pdf, show=params.show)

    print(f"Done. Wrote: {out_csv}, {out_pdf}, {out_params}")


if __name__ == "__main__":
    main()