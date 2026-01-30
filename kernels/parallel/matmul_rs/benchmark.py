import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path


gpu = os.environ.get("GPU", "")
assert gpu == "B200" or gpu == "H100", "GPU must be set to B200 or H100"

import torch

from common import (
    init_distributed_environment,
    destroy_distributed_environment,
    check_diff,
    benchmark_l2_clear,
    benchmark_no_l2_clear,
    profile,
    clean_print
)

from _C import TKParallelTensor, matmul_reduce_scatter  # type: ignore


def nccl_matmul_reduce_scatter_func(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> None:
    intermediate = torch.matmul(A, B if gpu == "H100" else B.T)
    torch.distributed.reduce_scatter_tensor(C, intermediate, op=torch.distributed.ReduceOp.SUM)


def tk_matmul_reduce_scatter_func(
    A: torch.Tensor,
    B: torch.Tensor,
    C: TKParallelTensor,
    barrier: TKParallelTensor,
) -> None:
    matmul_reduce_scatter(A, B, C, barrier)


def run(
    M: int,
    K: int,
    N: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
    check_correctness: bool = False,
    do_profile: bool = False,
    record_list_rank0: list | None = None, 
) -> None:
    A = torch.randn(M, K, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25
    if gpu == "H100":
        B = torch.randn(K, N, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25
    elif gpu == "B200":
        B = torch.randn(N, K, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25
    C_tk = TKParallelTensor(
        (M // local_world_size, N),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False
    )
    C_tk.data_.zero_()
    barrier = TKParallelTensor(
        (1, 1),
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    barrier.data_.zero_()

    C_nccl = torch.zeros(M // local_world_size, N, dtype=torch.bfloat16, device=f"cuda:{local_rank}")

    # Must wait for all barriers to be initialized before running the kernel
    torch.distributed.barrier()

    tk_run = lambda: tk_matmul_reduce_scatter_func(A, B, C_tk, barrier)
    nccl_run = lambda: nccl_matmul_reduce_scatter_func(A, B, C_nccl)

    if check_correctness:
        nccl_run()
        tk_run()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        check_diff("Matmul Reduce-Scatter Diff Comparison", C_tk.data_, C_nccl)

    if do_profile:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        combined = lambda: (nccl_run(), tk_run())
        profile(combined, num_iters)

    tk_avg_ms = benchmark_no_l2_clear(tk_run, num_warmup_iters, num_iters)
    nccl_avg_ms = benchmark_no_l2_clear(nccl_run, num_warmup_iters, num_iters)

    total_flops = 2.0 * M * N * K
    total_tflops = total_flops * 1e-12

    tk_tflops = total_tflops / (tk_avg_ms * 1e-3)
    nccl_tflops = total_tflops / (nccl_avg_ms * 1e-3)

    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<BF16 TP Matmul | world_size={local_world_size} | {M}x{K}x{N}>", print_once=True)
    clean_print(f"NCCL: {nccl_avg_ms:.3f} ms | {nccl_tflops:.2f} TFLOp/s")
    clean_print(f"PK: {tk_avg_ms:.3f} ms | {tk_tflops:.2f} TFLOp/s")
    payload = {
        "rank": local_rank,
        "world_size": local_world_size,
        "M": M, "K": K, "N": N,
        "nccl_ms": float(nccl_avg_ms),
        "nccl_tflops": float(nccl_tflops),
        "tk_ms": float(tk_avg_ms),
        "tk_tflops": float(tk_tflops),
    }

    gathered = [None for _ in range(local_world_size)]
    torch.distributed.all_gather_object(gathered, payload)

    if local_rank == 0 and record_list_rank0 is not None:
        record_list_rank0.extend(gathered)

def _maybe_import_plot_deps():
    # 只在 rank0 且需要导出时导入，避免每个 rank 都 import pandas/matplotlib
    import pandas as pd  # noqa
    import matplotlib.pyplot as plt  # noqa
    return pd, plt
def export_csv_and_plots(records: list, outdir: Path, prefix: str):
    pd, plt = _maybe_import_plot_deps()
    outdir.mkdir(parents=True, exist_ok=True)

    # 展开为 long-form：每条记录变成两行（impl=TK/NCCL）
    rows = []
    for r in records:
        rows.append({
            "rank": r["rank"], "world_size": r["world_size"],
            "M": r["M"], "K": r["K"], "N": r["N"],
            "impl": "NCCL", "ms": r["nccl_ms"], "tflops": r["nccl_tflops"]
        })
        rows.append({
            "rank": r["rank"], "world_size": r["world_size"],
            "M": r["M"], "K": r["K"], "N": r["N"],
            "impl": "PK", "ms": r["tk_ms"], "tflops": r["tk_tflops"]
        })

    df = pd.DataFrame(rows)
    raw_csv = outdir / f"{prefix}_raw.csv"
    df.to_csv(raw_csv, index=False)

    # 聚合：按 (M,K,N,num_comm_sms,impl) 对 ranks 做统计
    gcols = ["world_size", "M", "K", "N", "impl"]
    summ = df.groupby(gcols).agg(
        ms_mean=("ms", "mean"),
        ms_std=("ms", "std"),
        ms_min=("ms", "min"),
        ms_max=("ms", "max"),
        tflops_mean=("tflops", "mean"),
        tflops_std=("tflops", "std"),
        tflops_min=("tflops", "min"),
        tflops_max=("tflops", "max"),
        ranks=("rank", "nunique"),
    ).reset_index()

    sum_csv = outdir / f"{prefix}_summary.csv"
    summ.to_csv(sum_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out_parse", help="where to write csv/png (rank0 only)")
    parser.add_argument("--prefix", type=str, default="tp_matmul", help="file prefix")
    args = parser.parse_args()
    local_rank, local_world_size = init_distributed_environment()
    records_rank0 = [] if local_rank == 0 else None

    for N in [2048, 4096, 8192, 16384, 32768]:
    # for N in [32768]:
        run(N, N // local_world_size, N,
            local_rank, local_world_size, 
            check_correctness=False, do_profile=False,
            record_list_rank0=records_rank0)
    torch.distributed.barrier()
    if local_rank == 0:
        export_csv_and_plots(records_rank0, Path(args.outdir), args.prefix)


    destroy_distributed_environment()
