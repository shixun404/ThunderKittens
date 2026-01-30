#!/usr/bin/env python3
# pick_best_num_comm_sms.py
#
# Usage:
#   python pick_best_num_comm_sms.py input.csv output.csv
#
# Default selection:
#   1) minimize ms_mean
#   2) tie-break: maximize tflops_mean
#
# Output contains: world_size, M, K, N, best_num_comm_sms, best_impl, best_ms_mean, best_tflops_mean

import sys
import pandas as pd

def main(inp: str, outp: str) -> None:
    df = pd.read_csv(inp)

    required = {"M", "K", "N", "num_comm_sms", "ms_mean", "tflops_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric columns are numeric (robust to accidental strings)
    for col in ["M", "K", "N", "num_comm_sms", "ms_mean", "tflops_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid key metrics
    df = df.dropna(subset=["M", "K", "N", "num_comm_sms", "ms_mean", "tflops_mean"])

    # Step 1: within each (M,K,N,num_comm_sms), pick best impl row (fastest ms_mean, tie: higher tflops_mean)
    df1 = (
        df.sort_values(
            by=["M", "K", "N", "num_comm_sms", "ms_mean", "tflops_mean"],
            ascending=[True, True, True, True, True, False],
            kind="mergesort",  # stable
        )
        .groupby(["M", "K", "N", "num_comm_sms"], as_index=False)
        .first()
    )

    # Step 2: across num_comm_sms for each (M,K,N), pick global best
    df2 = (
        df1.sort_values(
            by=["M", "K", "N", "ms_mean", "tflops_mean"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        .groupby(["M", "K", "N"], as_index=False)
        .first()
    )

    # Prepare output columns
    out_cols = []
    if "world_size" in df2.columns:
        out_cols.append("world_size")

    out_df = df2.rename(
        columns={
            "num_comm_sms": "best_num_comm_sms",
            "impl": "best_impl",
            "ms_mean": "best_ms_mean",
            "tflops_mean": "best_tflops_mean",
        }
    )

    out_cols += ["M", "K", "N", "best_num_comm_sms"]
    if "best_impl" in out_df.columns:
        out_cols += ["best_impl", "best_ms_mean", "best_tflops_mean"]
    else:
        out_cols += ["best_ms_mean", "best_tflops_mean"]

    out_df = out_df[out_cols].sort_values(by=["M", "K", "N"]).reset_index(drop=True)
    out_df.to_csv(outp, index=False)
    print(out_df)
    print(f"Wrote {len(out_df)} rows to {outp}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pick_best.py tp_matmul_summary.csv ag_gemm_summary.csv", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])