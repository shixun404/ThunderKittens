#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Bar plot: fixed num_comm_sms, x=problem size, NCCL vs PK"
    )
    parser.add_argument("--csv", type=Path,
                        default="tp_matmul_summary.csv")
    # parser.add_argument("--num_comm_sms", type=int, required=True,
    #                     help="fixed num_comm_sms value")
    parser.add_argument("--metric", type=str, default="tflops",
                        choices=["tflops", "ms"],
                        help="metric to plot")
    parser.add_argument("--out", type=Path, default=Path("bar_plot.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # print(df)

    metric_col = "tflops_mean" if args.metric == "tflops" else "ms_mean"
    ylabel = "Mean TFLOP/s" if args.metric == "tflops" else "Mean Latency (ms)"
    

    impls = ["NCCL", "Triton-dist",  "PK"]

    orange = "#F29441"
    purple = "#9467BD"
    green = "#4EAE4E"
    green_blue = "#67BBBD"
    red = "#E67C7C"

    colors = {"NCCL": purple,  "Triton-dist":orange, "PK":green,}


    if df.empty:
        raise ValueError(f"No data")

    # 定义 problem size（你这里是 cubic：M=K=N*world_size）
    # 横轴直接用 logical N（输出矩阵列数）
    df["problem_size"] = df["N"]

    # 排序
    df = df.sort_values("problem_size")

    sizes = df["problem_size"].unique()
    print(sizes, df)

    # bar 布局
    bar_width = 0.8 / len(impls)
    x = range(len(sizes))

    plt.figure(figsize=(10, 3))

    for i, impl in enumerate(impls):
        sub = df[df["impl"] == impl].copy()
        print(sub)
        sub = sub.set_index("problem_size").loc[sizes]
        # print(metric_col, x)
        plt.bar(
            [xi + (i - 0.5) * bar_width for xi in x],
            sub[metric_col],
            width=bar_width,
            label=impl,
            color=colors[impl]
        )

        xs = [xi + (i - 0.5) * bar_width for xi in x]
        ys = sub[metric_col].values

        for x_pos, y_val in zip(xs, ys):
            plt.text(
                x_pos,
                y_val,
                f"{int(y_val)}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.xticks(list(x), [str(s) for s in sizes])
    plt.xlabel("Matrix Size, M=N=K")
    plt.ylabel(ylabel)
    plt.title(
        f"GEMM + RS, TFLOPS vs Problem Size"
    )
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, 850)
    plt.tight_layout()
    plt.savefig(f"bar_plot_GEMM_RS.png", dpi=200)
    plt.close()

    print(f"[OK] saved bar plot to bar_plot_GEMM_RS.png")


if __name__ == "__main__":
    main()
