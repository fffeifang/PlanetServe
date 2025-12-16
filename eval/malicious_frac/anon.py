import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from math import comb
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def h2(p: float) -> float:
    """Helper: -p*log2(p), safe for p<=0."""
    if p <= 0.0:
        return 0.0
    return -p * math.log2(p)


def write_csv(path: str, rows: List[Dict[str, float]], header: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.10g}" for k in header) + "\n")


# -----------------------------
# Models
# -----------------------------
def entropy_based_source_anonymity_gc(n: int, L: int, f: float, N: int = 10000, prob_eps: float = 1e-15) -> float:
    """
    Garlic Cast model: malicious nodes can collude across different paths.

    T = n*L total relay positions across all paths.
    s = number of malicious nodes among these positions ~ Binomial(T, f).
    We compute E[H_scenario] and normalize by log2(N).

    Returns: normalized anonymity in [0,1].
    """
    T = n * L
    logN = math.log2(N)
    H_expected = 0.0

    for s in range(T + 1):
        prob_s = comb(T, s) * (f ** s) * ((1 - f) ** (T - s))
        if prob_s < prob_eps:
            continue

        if s == 0:
            H_scenario = logN
        elif s == T:
            H_scenario = 0.0
        else:
            denom = float(T - s)
            if denom <= 0:
                H_scenario = 0.0
            else:
                sum_gamma = float(n) / denom
                p_pred = sum_gamma / n

                ohc = int(round((1.0 - f) * N)) - n
                if ohc < 1:
                    ohc = 1

                sum_others = 1.0 - sum_gamma
                p_other = sum_others / ohc

                H_pred = n * h2(p_pred)
                H_others = ohc * h2(p_other)
                H_scenario = H_pred + H_others

        H_expected += prob_s * H_scenario

    return H_expected / logN


def entropy_based_source_anonymity_planetserve(n: int, L: int, f: float, N: int = 10000, prob_eps: float = 1e-18) -> float:
    """
    PlanetServe model: malicious nodes only collude within the same path.
    A path is "compromised" if it has >=1 malicious relay.

    Enumerate all (s_1,...,s_n) where s_i ~ Binomial(L, f) independently, and compute
    E[H_scenario] normalized by log2(N).

    Returns: normalized anonymity in [0,1].
    """
    logN = math.log2(N)
    H_expected = 0.0

    # Precompute per-path pmf
    path_pmf = [comb(L, s_i) * (f ** s_i) * ((1 - f) ** (L - s_i)) for s_i in range(L + 1)]

    def recurse(path_index: int, compromised_paths: int, total_malicious: int, prob_so_far: float):
        nonlocal H_expected
        if path_index == n:
            if compromised_paths == 0:
                H_scenario = logN
            else:
                denom = float(n * L - total_malicious)
                if denom <= 0:
                    H_scenario = 0.0
                else:
                    sum_gamma = compromised_paths / denom
                    p_pred = sum_gamma / compromised_paths

                    ohc = int(round((1.0 - f) * N)) - compromised_paths
                    if ohc < 1:
                        ohc = 1

                    sum_others = 1.0 - sum_gamma
                    p_other = sum_others / ohc

                    H_pred = compromised_paths * h2(p_pred)
                    H_others = ohc * h2(p_other)
                    H_scenario = H_pred + H_others

            H_expected += prob_so_far * H_scenario
            return

        for s_i, p_i in enumerate(path_pmf):
            if p_i < prob_eps:
                continue
            new_prob = prob_so_far * p_i
            new_comp = compromised_paths + (1 if s_i > 0 else 0)
            new_tot_m = total_malicious + s_i
            recurse(path_index + 1, new_comp, new_tot_m, new_prob)

    recurse(0, 0, 0, 1.0)
    return H_expected / logN


# -----------------------------
# Plotting
# -----------------------------
def plot_curves(f_vals: List[float], series: Dict[str, List[float]], out_pdf: str, show: bool):
    colors = ['#15498D', '#E63660', '#83CBEB']
    markers = ['o', 's', '^']
    fontsize_axes = 10
    fontsize_ticks = 8
    fontsize_legend = 8
    linewidth = 1
    markeredgewidth = 1.02

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.set_position([0.15, 0.25, 0.8, 0.68])

    keys = list(series.keys())
    for i, k in enumerate(keys):
        ax.plot(
            f_vals, series[k],
            label=k,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=linewidth,
            markeredgewidth=markeredgewidth,
            markerfacecolor='none',
            linestyle='-',
            clip_on=False
        )

    ax.set_xlabel('Fraction of malicious nodes (f)', fontsize=fontsize_axes)
    ax.set_ylabel('Anonymity (normalized entropy)', fontsize=fontsize_axes)

    ax.set_ylim(0, 1)
    ax.set_xlim(min(f_vals), max(f_vals))

    ax.tick_params(axis='x', labelsize=fontsize_ticks)
    ax.tick_params(axis='y', labelsize=fontsize_ticks)

    # y ticks
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))


    ax.legend(loc='lower left', bbox_to_anchor=(0.1, 0.15),
              fontsize=fontsize_legend, frameon=True, edgecolor='black')

    fig.savefig(out_pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
@dataclass
class Scheme:
    name: str
    n: int
    L: int
    model: str  # "gc" or "ps"


def parse_f_values(s: str) -> List[float]:
    # Allow comma-separated list
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if any(v <= 0 or v >= 1 for v in vals):
        raise ValueError("All f values must be in (0,1).")
    return vals


def main():
    ap = argparse.ArgumentParser(description=" entropy-based anonymity plotter.")
    ap.add_argument("--N", type=int, default=10000, help="Total number of nodes N")
    ap.add_argument("--f-values", type=str,
                    default="0.001,0.01,0.05,0.1,0.2,0.3,0.5",
                    help="Comma-separated f values (subset used for plotting).")
    ap.add_argument("--output-dir", type=str, default="out")
    ap.add_argument("--show", action="store_true", help="Show plot (off by default for headless env).")
    ap.add_argument("--prob-eps", type=float, default=1e-15, help="Skip probability terms below this threshold.")
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    f_vals = parse_f_values(args.f_values)

    schemes = [
        Scheme(name="PlanetServe", n=4, L=4, model="ps"),
        Scheme(name="Garlic Cast", n=4, L=4, model="gc"),
        Scheme(name="Onion", n=1, L=5, model="ps"),  # n=1 => PS/GC identical; PS function is fine
    ]

    series: Dict[str, List[float]] = {s.name: [] for s in schemes}

    for f in f_vals:
        for s in schemes:
            if s.model == "gc":
                a = entropy_based_source_anonymity_gc(s.n, s.L, f, N=args.N, prob_eps=args.prob_eps)
            elif s.model == "ps":
                a = entropy_based_source_anonymity_planetserve(s.n, s.L, f, N=args.N, prob_eps=max(args.prob_eps, 1e-18))
            else:
                raise ValueError(f"Unknown model: {s.model}")
            series[s.name].append(a)

    # export CSV
    rows = []
    for i, f in enumerate(f_vals):
        r = {"f": f}
        for k in series:
            r[k] = float(series[k][i])
        rows.append(r)

    out_csv = os.path.join(args.output_dir, "anonymity.csv")
    header = ["f"] + [s.name for s in schemes]
    write_csv(out_csv, rows, header)

    # export params
    meta = {
        "N": args.N,
        "f_values": f_vals,
        "schemes": [asdict(s) for s in schemes],
        "prob_eps": args.prob_eps,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": {
            "csv": "anonymity.csv",
            "pdf": "anonymity.pdf"
        }
    }
    with open(os.path.join(args.output_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # plot PDF
    out_pdf = os.path.join(args.output_dir, "anonymity.pdf")
    plot_curves(f_vals, series, out_pdf, show=args.show)

    print(f"Done. Wrote: {out_csv}, {out_pdf}, {os.path.join(args.output_dir,'params.json')}")


if __name__ == "__main__":
    main()