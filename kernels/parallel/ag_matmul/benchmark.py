import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
from pathlib import Path

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

from _C import TKParallelTensor, all_gather_matmul  # type: ignore


def nccl_all_gather_matmul_func(
    A_local: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> None:
    torch.distributed.all_gather_into_tensor(A, A_local)
    torch.matmul(A, B, out=C)


def tk_all_gather_matmul_func(
    A: TKParallelTensor,
    B: torch.Tensor,
    C: torch.Tensor,
    barrier: TKParallelTensor,
    num_comm_sms: int
) -> None:
    all_gather_matmul(A, B, C, barrier, num_comm_sms)


def run(
    M: int,
    K: int,
    N: int,
    num_comm_sms: int,
    local_rank: int,
    local_world_size: int,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
    check_correctness: bool = False,
    do_profile: bool = False,
    record_list_rank0: list | None = None, 
) -> None:
    A_local = torch.randn(M // local_world_size, K, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25
    A_tk = TKParallelTensor(
        (M, K),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    # Note: this is ONLY for convenience; separating A and A_local does NOT affect performance
    A_tk.data_[local_rank * (M // local_world_size):(local_rank + 1) * (M // local_world_size)] = A_local
    A_nccl = torch.zeros(M, K, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    B = torch.randn(K, N, dtype=torch.bfloat16, device=f"cuda:{local_rank}") / K ** 0.25
    C_tk = torch.zeros(M, N, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    C_nccl = torch.zeros(M, N, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    barrier = TKParallelTensor(
        (2, 1024, 1024), # 2 MB is the minimum requirement for multicast object anyways, so no loss here!
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True
    )
    barrier.data_.zero_()

    # Must wait for all barriers to be initialized before running the kernel
    torch.distributed.barrier()

    tk_run = lambda: tk_all_gather_matmul_func(A_tk, B, C_tk, barrier, num_comm_sms)
    nccl_run = lambda: nccl_all_gather_matmul_func(A_local, A_nccl, B, C_nccl)

    if check_correctness:
        nccl_run()
        tk_run()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        check_diff("All-Gather Matmul Diff Comparison", C_tk, C_nccl)

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
    clean_print(f"<BF16 TP Matmul | world_size={local_world_size} | {M}x{K}x{N} | num_comm_sms={num_comm_sms}>", print_once=True)
    clean_print(f"NCCL: {nccl_avg_ms:.3f} ms | {nccl_tflops:.2f} TFLOp/s")
    clean_print(f"TK: {tk_avg_ms:.3f} ms | {tk_tflops:.2f} TFLOp/s")

    payload = {
        "rank": local_rank,
        "world_size": local_world_size,
        "M": M, "K": K, "N": N,
        "num_comm_sms": num_comm_sms,
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
            "M": r["M"], "K": r["K"], "N": r["N"], "num_comm_sms": r["num_comm_sms"],
            "impl": "NCCL", "ms": r["nccl_ms"], "tflops": r["nccl_tflops"]
        })
        rows.append({
            "rank": r["rank"], "world_size": r["world_size"],
            "M": r["M"], "K": r["K"], "N": r["N"], "num_comm_sms": r["num_comm_sms"],
            "impl": "TK", "ms": r["tk_ms"], "tflops": r["tk_tflops"]
        })

    df = pd.DataFrame(rows)
    raw_csv = outdir / f"{prefix}_raw.csv"
    df.to_csv(raw_csv, index=False)

    # 聚合：按 (M,K,N,num_comm_sms,impl) 对 ranks 做统计
    gcols = ["world_size", "M", "K", "N", "num_comm_sms", "impl"]
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--profile", action="store_true")
    # 允许你从命令行覆盖 sweep
    parser.add_argument("--Ns", type=str, default="2048,4096,8192,16384,32768")
    parser.add_argument("--comm_sms", type=str, default="1,2,4,8,16,32,64")
    # parser.add_argument("--Ns", type=str, default="32768")
    # parser.add_argument("--comm_sms", type=str, default="16")
    # parser.add_argument("--comm_sms", type=str, default="1,2,4,8,16,32,64")
    args = parser.parse_args()

    local_rank, local_world_size = init_distributed_environment()

    records_rank0 = [] if local_rank == 0 else None

    Ns = [int(x) for x in args.Ns.split(",") if x.strip()]
    comm_sms_list = [int(x) for x in args.comm_sms.split(",") if x.strip()]
  
    for N in Ns:
        for num_comm_sms in comm_sms_list:
            # 你原来 run(N, N, N // world_size, ...) 的含义是：M=N, K=N, N_out=N/world_size
            run(
                N, N, N // local_world_size,
                num_comm_sms,
                local_rank, local_world_size,
                num_warmup_iters=args.warmup,
                num_iters=args.iters,
                check_correctness=args.check,
                do_profile=args.profile,
                record_list_rank0=records_rank0
            )

    torch.distributed.barrier()
    if local_rank == 0:
        export_csv_and_plots(records_rank0, Path(args.outdir), args.prefix)

    destroy_distributed_environment()
