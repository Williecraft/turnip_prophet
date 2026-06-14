#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset.py — 批次生成「與官方完全一致」的大頭菜整週價格資料。
Generate a dataset of ACNH turnip weeks for later strategy mining.

每一列 = 一週。欄位刻意為「下一階段策略推導」設計:
    week_index     : 在序列中的第幾週 (chain 模式下相鄰列即相鄰週, 可做跨週分析)
    seed           : 用來產生此週的 32-bit 種子 (可重現)
    prev_pattern   : 上週 pattern (影響本週 pattern 機率)
    base_price     : 週日 Daisy Mae 買價 (90~110)
    pattern        : 本週 pattern (0~3)
    price_Mon_AM .. price_Sat_PM : 12 個賣價時段
    max_price      : 本週最高賣價
    max_price_slot : 最高價落在第幾個時段 (0=Mon_AM .. 11=Sat_PM)
    peak_multiplier: max_price / base_price (報酬率上限)

跨週只有「波型」會傳遞, 價格不會 (memoryless): 官方規則中上週波型 (whatPattern) 透過轉移表
影響本週波型, 但上週價格對本週價格毫無影響, base_price 每週重抽。因此每週獨立生成、把上週
波型當成條件欄位 (prev_pattern) 就足夠且正確 —— 這也是預設的 independent 模式。

兩種模式 (mode):
  independent (預設)  : 每週獨立。prev_pattern 由 --prev-pattern 指定 (預設均勻隨機 0~3,
                        讓四種「上週波型」條件都有充足樣本, 最利於學條件式買賣規則)。
  chain (選用)        : 每週結果波型餵給下週當 prev_pattern, 形成連鎖, 產生官方長期
                        (stationary)邊際波型分布。僅在想估「整體遇到各波型頻率」時才需要。

用法範例 (examples):
    # 100 萬週獨立 + 上週波型均勻抽 (預設), 輸出 parquet
    python generate_dataset.py --n 1000000 --out weeks.parquet

    # 只研究「上週 = large_spike(1)」之後一週的分布
    python generate_dataset.py --n 200000 --prev-pattern 1

    # 連鎖模式 (長期邊際分布), 連續枚舉種子可重現
    python generate_dataset.py --n 500000 --mode chain --seed-start 0

    # 輸出 CSV
    python generate_dataset.py --n 100000 --format csv --out weeks.csv
"""

import argparse
import random
import sys

from turnip_sim import generate_week, DAY_LABELS

PRICE_COLS = [f"price_{lbl}" for lbl in DAY_LABELS]
COLUMNS = (
    ["week_index", "seed", "prev_pattern", "base_price", "pattern"]
    + PRICE_COLS
    + ["max_price", "max_price_slot", "peak_multiplier"]
)


def _make_row(week_index, seed, prev, w):
    prices = w["prices"]
    base = w["base_price"]
    mx = max(prices)
    row = {
        "week_index": week_index,
        "seed": seed,
        "prev_pattern": prev,
        "base_price": base,
        "pattern": w["pattern"],
        "max_price": mx,
        "max_price_slot": prices.index(mx),
        "peak_multiplier": mx / base,
    }
    for col, v in zip(PRICE_COLS, prices):
        row[col] = v
    return row


def iter_rows(n, mode, seed_start, prev_pattern_arg, start_pattern, burn_in, rng):
    """產生 n 列 (欄位見 COLUMNS)。

    mode == "chain"       : 連鎖。prev = 上一週的結果 pattern; 起始 prev = start_pattern,
                            先跑 burn_in 週讓初始條件淡出 (不寫出)。
    mode == "independent" : 獨立。prev = prev_pattern_arg (固定) 或每週均勻隨機。
    seed_start            : 連續枚舉種子 (可重現); None 則用隨機 32-bit 種子。
    兩模式的暖機週同樣會消耗種子序列, 以維持可重現性。
    """
    counter = 0  # 已消耗的種子序數 (含暖機), 確保 seed 唯一且可重現

    def next_seed():
        nonlocal counter
        s = (seed_start + counter) & 0xFFFFFFFF if seed_start is not None else rng.getrandbits(32)
        counter += 1
        return s

    if mode == "chain":
        prev = start_pattern
        for _ in range(burn_in):          # 暖機: 推進連鎖但不輸出
            prev = generate_week(prev, next_seed())["pattern"]
        for w_idx in range(n):
            seed = next_seed()
            w = generate_week(prev, seed)
            yield _make_row(w_idx, seed, prev, w)
            prev = w["pattern"]           # 本週結果 -> 下週的 prev
    else:  # independent
        for w_idx in range(n):
            seed = next_seed()
            prev = prev_pattern_arg if prev_pattern_arg is not None else rng.randint(0, 3)
            yield _make_row(w_idx, seed, prev, generate_week(prev, seed))


def write_parquet(rows_iter, out_path, batch_size):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("需要 pyarrow 才能輸出 parquet。請 `pip install pyarrow`, 或改用 --format csv。")

    schema = pa.schema(
        [("week_index", pa.int64()), ("seed", pa.uint32()), ("prev_pattern", pa.int8()),
         ("base_price", pa.int16()), ("pattern", pa.int8())]
        + [(c, pa.int32()) for c in PRICE_COLS]
        + [("max_price", pa.int32()), ("max_price_slot", pa.int8()),
           ("peak_multiplier", pa.float32())]
    )

    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    total = 0
    batch = {c: [] for c in COLUMNS}
    try:
        for row in rows_iter:
            for c in COLUMNS:
                batch[c].append(row[c])
            if len(batch["seed"]) >= batch_size:
                writer.write_table(pa.table(batch, schema=schema))
                total += len(batch["seed"])
                batch = {c: [] for c in COLUMNS}
        if batch["seed"]:
            writer.write_table(pa.table(batch, schema=schema))
            total += len(batch["seed"])
    finally:
        writer.close()
    return total


def write_csv(rows_iter, out_path):
    import csv

    total = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows_iter:
            writer.writerow(row)
            total += 1
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成 ACNH 大頭菜整週價格資料集")
    ap.add_argument("--n", type=int, required=True, help="要生成幾週 (列數)")
    ap.add_argument("--out", default=None, help="輸出檔名 (預設 weeks.<格式副檔名>)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--mode", choices=["chain", "independent"], default="independent",
                    help="independent(預設,每週獨立+上週波型當條件) 或 chain(連鎖,長期邊際分布)")
    ap.add_argument("--seed-start", type=int, default=None,
                    help="從此種子起連續枚舉 (可重現); 不給則用隨機 32-bit 種子")
    ap.add_argument("--prev-pattern", default="random",
                    help="[independent 模式] 上週 pattern: 0/1/2/3 固定, 或 'random' (均勻抽 0~3)")
    ap.add_argument("--start-pattern", type=int, default=2,
                    help="[chain 模式] 連鎖起始 pattern (預設 2, 同官方未初始化)")
    ap.add_argument("--burn-in", type=int, default=50,
                    help="[chain 模式] 暖機週數, 讓初始條件淡出後才開始輸出 (預設 50)")
    ap.add_argument("--rng-seed", type=int, default=None,
                    help="控制『隨機種子/隨機上週pattern』的亂數 (讓整批可重現)")
    ap.add_argument("--batch-size", type=int, default=100_000, help="parquet 分批寫入大小")
    args = ap.parse_args(argv)

    if args.prev_pattern == "random":
        prev_arg = None
    else:
        prev_arg = int(args.prev_pattern)
        if prev_arg not in (0, 1, 2, 3):
            ap.error("--prev-pattern 必須是 0/1/2/3 或 random")
    if args.start_pattern not in (0, 1, 2, 3):
        ap.error("--start-pattern 必須是 0/1/2/3")

    out = args.out or f"weeks.{ 'parquet' if args.format == 'parquet' else 'csv' }"
    rng = random.Random(args.rng_seed)

    rows = iter_rows(args.n, args.mode, args.seed_start, prev_arg,
                     args.start_pattern, args.burn_in, rng)
    if args.format == "parquet":
        total = write_parquet(rows, out, args.batch_size)
    else:
        total = write_csv(rows, out)

    print(f"[OK] 已寫出 {total} 週 -> {out} ({args.format})")


if __name__ == "__main__":
    main()
