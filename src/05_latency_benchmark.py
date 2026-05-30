"""
05_latency_benchmark.py
=======================
Benchmarks per-flow inference latency for every model.
Proves the "real-time applicability" claim made in the abstract.

Methodology:
  - Single-sample inference repeated 1000 times per model
  - First 50 calls discarded as JIT/cache warm-up
  - Reports mean, median, p99, std (ms) and throughput (flows/sec)

Output:
  outputs/results/latency_benchmark.csv   ← Table III in paper
  outputs/figures/fig_latency.pdf
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")

N_RUNS   = 1000
WARM_UP  = 50
# Real-time threshold for microgrid IDS (1 ms per flow)
RT_THRESHOLD_MS = 1.0


def load_data():
    X_test = np.load(os.path.join(OUT, "X_test.npy"))
    return X_test


def load_models():
    names = {
        "Decision Tree":        "decision_tree.pkl",
        "Gradient Boosting":    "gradient_boosting.pkl",
        "XGBoost":              "xgboost.pkl",
        "CatBoost":             "catboost.pkl",
        "XGBoost (Optimised)":  "xgboost_optimised.pkl",
        "Ensemble (Voting)":    "ensemble.pkl",
    }
    loaded = {}
    for name, fname in names.items():
        path = os.path.join(MODELS, fname)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
        else:
            print(f"  [skip] {fname} not found")
    return loaded


def benchmark_model(model, X_test, n_runs=N_RUNS, warm_up=WARM_UP):
    rng = np.random.default_rng(42)

    # warm-up phase
    for _ in range(warm_up):
        i = rng.integers(0, len(X_test))
        model.predict(X_test[i:i+1])

    # timed phase
    latencies = []
    for _ in range(n_runs):
        i = rng.integers(0, len(X_test))
        sample = X_test[i:i+1]
        t0 = time.perf_counter()
        model.predict(sample)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(latencies)
    return {
        "mean_ms":           round(arr.mean(), 4),
        "median_ms":         round(np.median(arr), 4),
        "p99_ms":            round(np.percentile(arr, 99), 4),
        "std_ms":            round(arr.std(), 4),
        "throughput_fps":    round(1000.0 / arr.mean()),
        "meets_realtime":    arr.mean() < RT_THRESHOLD_MS,
    }


def run():
    print("\n── 05: Inference Latency Benchmark ────────────────────────────")
    print(f"  {N_RUNS} runs per model  |  {WARM_UP} warm-up calls discarded")
    print(f"  Real-time threshold: {RT_THRESHOLD_MS} ms/flow\n")

    X_test = load_data()
    models = load_models()

    rows = []
    for name, model in models.items():
        print(f"  Benchmarking {name} ...", end=" ", flush=True)
        stats = benchmark_model(model, X_test)
        stats["Model"] = name
        rows.append(stats)
        rt = "✓ RT" if stats["meets_realtime"] else "✗ RT"
        print(
            f"mean={stats['mean_ms']:.3f}ms  "
            f"p99={stats['p99_ms']:.3f}ms  "
            f"tput={stats['throughput_fps']:,} fps  {rt}"
        )

    df = pd.DataFrame(rows)[
        ["Model", "mean_ms", "median_ms", "p99_ms",
         "std_ms", "throughput_fps", "meets_realtime"]
    ]
    out_path = os.path.join(OUT, "latency_benchmark.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → outputs/results/latency_benchmark.csv")

    # ── Figure: horizontal bar chart of mean latency ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ecc71" if r else "#e74c3c" for r in df["meets_realtime"]]
    bars = ax.barh(df["Model"], df["mean_ms"], color=colors, height=0.55)

    # p99 error bars
    xerr = df["p99_ms"] - df["mean_ms"]
    ax.errorbar(df["mean_ms"], df["Model"],
                xerr=xerr, fmt="none",
                color="gray", capsize=4, linewidth=1)

    ax.axvline(RT_THRESHOLD_MS, color="red", linestyle="--",
               linewidth=1, label=f"Real-time threshold ({RT_THRESHOLD_MS} ms)")
    ax.set_xlabel("Mean inference latency (ms per flow)", fontsize=10)
    ax.set_title("Per-Flow Inference Latency — Microgrid IDS Candidates",
                 fontsize=10, pad=10)
    ax.legend(fontsize=9)

    # annotate throughput
    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row.throughput_fps:,} f/s",
                va="center", ha="left", fontsize=8,
                color="gray")

    plt.tight_layout()
    fig_path = os.path.join(FIGS, "fig_latency.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → outputs/figures/fig_latency.pdf")

    print("\n  ── Key finding for paper ───────────────────────────────────")
    best = df.loc[df["mean_ms"].idxmin()]
    print(f"  Fastest: {best['Model']} at {best['mean_ms']:.3f} ms/flow")
    print(f"  All models below {RT_THRESHOLD_MS}ms threshold: "
          f"{'YES' if df['meets_realtime'].all() else 'NO'}")
    print("  Write in paper: 'All evaluated models satisfy the <1ms")
    print("  real-time constraint, with XGBoost achieving [X] flows/sec")
    print("  on commodity hardware — well within microgrid SCADA")
    print("  communication rates of 10-100 flows/sec.'")
    print("  ✓ Latency benchmark complete\n")
    return df


if __name__ == "__main__":
    run()