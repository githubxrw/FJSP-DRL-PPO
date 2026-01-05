import os
import re
import glob
import hashlib
import time
import numpy as np
import pandas as pd


def _safe_sheet_name(base: str, suffix: str) -> str:
    """
    Excel sheet 名称限制：<=31 字符，且不能包含 : \ / ? * [ ]
    这里做：清洗 + 截断 + 加短hash保证唯一性
    """
    base = re.sub(r'[:\\/?*\[\]]', '_', base)
    base = re.sub(r'\s+', '_', base).strip('_')
    raw = f"{base}_{suffix}"

    if len(raw) <= 31:
        return raw

    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:6]
    # 预留：_ + suffix + _ + hash 共 (1+len(suffix)+1+6) 字符
    reserve = 1 + len(suffix) + 1 + 6
    keep = 31 - reserve
    raw_trunc = f"{raw[:keep]}_{suffix}_{h}"
    return raw_trunc[:31]


def _load_npy_to_mk_time_tables(npy_path: str):
    """
    支持两类输入：
    1) arr.shape = [N, 2]              -> 每个 instance 1 次结果
    2) arr.shape = [N, R, 2]           -> 每个 instance R 次结果（如100次）

    返回：
        mk_df: index=inst_****, columns=run_*** 或 mk
        t_df : index=inst_****, columns=run_*** 或 time
    """
    arr = np.load(npy_path, allow_pickle=False)

    if arr.ndim == 2 and arr.shape[1] == 2:
        # [N, 2]
        mk = arr[:, 0].reshape(-1, 1)
        tm = arr[:, 1].reshape(-1, 1)
        cols_mk = ["mk"]
        cols_tm = ["time"]

    elif arr.ndim == 3 and arr.shape[2] == 2:
        # [N, R, 2]
        mk = arr[:, :, 0]
        tm = arr[:, :, 1]
        R = mk.shape[1]
        cols_mk = [f"run_{r:03d}" for r in range(R)]
        cols_tm = [f"run_{r:03d}" for r in range(R)]

    else:
        raise ValueError(
            f"Unsupported npy shape: {arr.shape} in {npy_path}. "
            f"Expect [N,2] or [N,R,2]."
        )

    N = mk.shape[0]
    inst_index = [f"inst_{i:04d}" for i in range(N)]

    mk_df = pd.DataFrame(mk, index=inst_index, columns=cols_mk)
    t_df = pd.DataFrame(tm, index=inst_index, columns=cols_tm)
    return mk_df, t_df


def export_instances_to_excel(
    root_dir: str = "./test_results/BenchData",
    out_dir: str = "./TestDataToExcel/BenchData",
    out_name: str | None = None,
):
    """
    递归读取 root_dir 下所有 npy，并输出到一个 Excel：
      - 每个 npy 生成 2 个 sheet：makespan / time
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"root_dir not found: {root_dir}")

    os.makedirs(out_dir, exist_ok=True)

    if out_name is None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_name = f"instances_{ts}.xlsx"

    out_path = os.path.join(out_dir, out_name)

    npy_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True))
    if not npy_files:
        raise FileNotFoundError(f"No npy files under: {root_dir}")

    # 写 Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for fp in npy_files:
            # 用 “子目录名 + 文件名” 构建 sheet 基名，避免不同目录下同名文件冲突
            rel = os.path.relpath(fp, root_dir)  # e.g. Brandimarte/Result_xxx.npy
            rel_no_ext = os.path.splitext(rel)[0]
            base_name = rel_no_ext.replace(os.sep, "__")  # Brandimarte__Result_xxx

            mk_df, t_df = _load_npy_to_mk_time_tables(fp)

            sheet_mk = _safe_sheet_name(base_name, "mk")
            sheet_t = _safe_sheet_name(base_name, "t")

            mk_df.to_excel(writer, sheet_name=sheet_mk, index=True)
            t_df.to_excel(writer, sheet_name=sheet_t, index=True)

    print(f"[OK] Exported to: {out_path}")
    print(f"[Info] Total npy files: {len(npy_files)}")


if __name__ == "__main__":
    export_instances_to_excel(
        root_dir="./test_results/BenchData",
        out_dir="./TestDataToExcel/BenchData",
        out_name=None,   # 自动按时间命名
    )
