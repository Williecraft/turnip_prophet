#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webapp/precompute.py — 離線預訓練所有策略 → data/policies.pkl

Streamlit Cloud 免費方案冷啟動 + 每次互動都重訓策略會很慢, 故先把所有可能的
(買價 90~110, 上週波型 None/0/1/2/3) = 21 × 5 = 105 套策略離線訓練好序列化。
UI 啟動只需 pickle.load, 查表即得, 反應即時。把產出的 data/policies.pkl 一併 commit
進 repo, 雲端就完全不必訓練。

用法:
    python webapp/precompute.py            # 預設品質 (與 policy_dp 預設一致)
    python webapp/precompute.py --fast     # 較少情境 (除錯用, 品質略降)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORE = os.path.join(_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from policy_dp import train_policy  # noqa: E402

from advisor_api import BUY_PRICE_RANGE, PREV_PATTERNS, POLICIES_PKL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="較少情境 (除錯用)")
    ap.add_argument("--out", default=POLICIES_PKL, help="輸出 pkl 路徑")
    args = ap.parse_args()

    kw = dict(n_train=20000, n_eval=20000) if args.fast else {}
    keys = [(b, p) for p in PREV_PATTERNS for b in BUY_PRICE_RANGE]
    total = len(keys)
    print(f"預訓練 {total} 套策略 (買價 {min(BUY_PRICE_RANGE)}~{max(BUY_PRICE_RANGE)} "
          f"× 上週波型 {PREV_PATTERNS}){' [fast]' if args.fast else ''} ...")

    policies = {}
    t0 = time.time()
    for i, (base, prev) in enumerate(keys, 1):
        policies[(base, prev)] = train_policy(base, prev, **kw)
        if i % 10 == 0 or i == total:
            el = time.time() - t0
            print(f"  {i:>3}/{total}  ({el:5.1f}s, ~{el/i:4.2f}s/套)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(policies, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"完成: {args.out}  ({len(policies)} 套, {size_mb:.2f} MB, "
          f"共 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
