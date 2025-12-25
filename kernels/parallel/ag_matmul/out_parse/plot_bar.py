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

    df_ = pd.read_csv(args.csv)
    # print(df)

    metric_col = "tflops_mean" if args.metric == "tflops" else "ms_mean"
    ylabel = "Mean TFLOp/s" if args.metric == "tflops" else "Mean Latency (ms)"
    impls = ["NCCL", "Triton-dist", "PK"]
    
    orange = "#F29441"
    purple = "#9467BD"
    green = "#4EAE4E"
    green_blue = "#67BBBD"
    red = "#E67C7C"

    colors = {"NCCL": purple, "Triton-dist": orange, "PK":green}
    comm_list = [1, 2, 4, 8, 16, 32, 64]

    df = df_[df_["num_comm_sms"].isin(comm_list)].copy()

    idx = df.groupby(
        ["impl", "M", "K", "N"]  # ← 根据你的数据结构，可加 world_size
    )["tflops_mean"].idxmax()

    df_max = df.loc[idx].reset_index(drop=True)
    df = df_max
    # 固定 num_comm_sms
    # for num_comm_sms in [1]:
    # df = df_[df_["num_comm_sms"] == num_comm_sms].copy()
    if df.empty:
        raise ValueError(f"No data for num_comm_sms")

    # 定义 problem size（你这里是 cubic：M=K=N*world_size）
    # 横轴直接用 logical N（输出矩阵列数）
    df["problem_size"] = df["M"]

    # 排序
    df = df.sort_values("problem_size")

    sizes = df["problem_size"].unique()

    # bar 布局
    bar_width = 0.8 / len(impls)
    x = range(len(sizes))

    plt.figure(figsize=(10, 3))

    for i, impl in enumerate(impls):
        sub = df[df["impl"] == impl]
        sub = sub.set_index("problem_size").loc[sizes]

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
            plt.text(
                x_pos,
                y_val + 60,
                f"{int(100 * y_val / 989)}%",
                ha="center",
                va="bottom",
                fontsize=9
            )
    roofline_x = [i - 0.5 for i in range(len(sizes) + 1)]
    y = [989 for _ in roofline_x]
    plt.plot(roofline_x, y, color="black", linestyle="--", label="Roofline")
    plt.xticks(list(x), [str(s) for s in sizes])
    plt.xlabel("Matrx Size, M=N=K")
    plt.ylabel(ylabel)
    plt.title(
        f"AG+GEMM,  {ylabel} vs Problem Size"
    )
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    # plt.ylim([0, 850])
    plt.tight_layout()
    plt.savefig(f"bar_plot.png", dpi=200)
    plt.close()

    print(f"[OK] saved bar plot to bar_plot.png")


if __name__ == "__main__":
    main()
