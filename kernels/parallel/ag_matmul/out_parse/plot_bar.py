#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Bar plot: fixed num_comm_sms, x=problem size, NCCL vs TK"
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="out_parse/tp_matmul_summary.csv")
    parser.add_argument("--num_comm_sms", type=int, required=True,
                        help="fixed num_comm_sms value")
    parser.add_argument("--metric", type=str, default="tflops",
                        choices=["tflops", "ms"],
                        help="metric to plot")
    parser.add_argument("--out", type=Path, default=Path("bar_plot.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    metric_col = "tflops_mean" if args.metric == "tflops" else "ms_mean"
    ylabel = "Mean TFLOp/s" if args.metric == "tflops" else "Mean Latency (ms)"

    # 固定 num_comm_sms
    df = df[df["num_comm_sms"] == args.num_comm_sms].copy()
    if df.empty:
        raise ValueError(f"No data for num_comm_sms={args.num_comm_sms}")

    # 定义 problem size（你这里是 cubic：M=K=N*world_size）
    # 横轴直接用 logical N（输出矩阵列数）
    df["problem_size"] = df["N"]

    # 排序
    df = df.sort_values("problem_size")

    impls = ["NCCL", "TK"]
    sizes = df["problem_size"].unique()

    # bar 布局
    bar_width = 0.35
    x = range(len(sizes))

    plt.figure(figsize=(10, 5))

    for i, impl in enumerate(impls):
        sub = df[df["impl"] == impl]
        sub = sub.set_index("problem_size").loc[sizes]

        plt.bar(
            [xi + (i - 0.5) * bar_width for xi in x],
            sub[metric_col],
            width=bar_width,
            label=impl
        )

    plt.xticks(list(x), [str(s) for s in sizes])
    plt.xlabel("Problem Size (N)")
    plt.ylabel(ylabel)
    plt.title(
        f"Fixed num_comm_sms = {args.num_comm_sms} | {ylabel} vs Problem Size"
    )
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"[OK] saved bar plot to {args.out}")


if __name__ == "__main__":
    main()
