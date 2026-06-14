#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py — 大規模評估 (evaluate)

誠實原則: 策略是用 forecaster 的「子波型取樣器」訓練的, 這裡改用**獨立的 bit-exact 官方
模擬器 turnip_sim.generate_week** 產生測試週 (out-of-sample), 確認策略能類化且勝過基準線。

內容:
  1. 取樣器忠實度: 子波型取樣器的統計量 (各時段平均、波型分布) vs turnip_sim 一致。
  2. 策略比較表 (對每個上週波型, 買價=100): max_profit/winrate/cvar + 基準線
     (oracle 完美後見 / naive 週六硬賣 / greedy 門檻), 指標: 平均/中位/勝率/最壞/CVaR5。
  3. forecaster 校準: 預測波型機率 vs 實際命中率 (bucket)。
"""

from __future__ import annotations

import time
import numpy as np

from turnip_sim import generate_week, PATTERN_NAMES
from forecaster import forecast, enumerate_subpatterns, STATIONARY
from policy_dp import (train_policy, sample_scenarios, _apply_sell_rule,
                       _cvar, OBJECTIVES, N_SLOTS)

BASE = 100


def true_weeks(prev, base, n, rng):
    """用官方模擬器產 n 個 (此 prev 且 base==base) 的真實週 -> (n,12) int。"""
    out = np.empty((n, N_SLOTS), dtype=np.int64)
    got = 0
    while got < n:
        # prev=None (未知上週) -> 依 stationary 邊際抽一個真實 prev
        pv = int(rng.choice(4, p=STATIONARY)) if prev is None else prev
        w = generate_week(pv, int(rng.integers(0, 2**32)))
        if w["base_price"] == base:
            out[got] = w["prices"]
            got += 1
    return out


def metrics(U, base):
    R = U / base
    return {
        "mean%": (R.mean() - 1) * 100,
        "median%": (np.median(R) - 1) * 100,
        "win%": (U >= base).mean() * 100,
        "worst%": (R.min() - 1) * 100,
        "cvar5%": (_cvar(R) - 1) * 100,
    }


def greedy_sell(prices, base, k=1.3):
    """基準線: 第一次 price>=k*base 就賣, 否則最後一格。"""
    n, T = prices.shape
    sold = np.zeros(n, bool); U = np.zeros(n)
    for t in range(T):
        do = (~sold) & ((prices[:, t] >= k * base) | (t == T - 1))
        U[do] = prices[do, t]; sold |= do
    return U


# --------------------------------------------------------------------------
def check_sampler_fidelity(rng):
    print("=== 1. 取樣器忠實度 (subpattern sampler vs 官方 turnip_sim, prev=2,base=100) ===")
    n = 60000
    samp, pat = sample_scenarios(BASE, 2, n, rng)
    true = true_weeks(2, BASE, 20000, rng)
    print(f"  各時段平均價 (採樣 vs 官方), 最大絕對差 = "
          f"{np.abs(samp.mean(0) - true.mean(0)).max():.2f} 鈴")
    # 波型分布
    subs = enumerate_subpatterns(BASE, 2)
    print(f"  採樣 oracle 均值={samp.max(1).mean():.1f}  官方 oracle 均值={true.max(1).mean():.1f}")


def compare_policies(rng):
    print("\n=== 2. 策略比較 (測試於官方 turnip_sim 真實週, 買價=100) ===")
    for prev in (0, 1, 2, 3, None):
        pol = train_policy(BASE, prev, n_train=40000, n_eval=60000, seed=7)
        test = true_weeks(prev, BASE, 20000, rng)

        rows = {}
        rows["oracle"] = metrics(_apply_sell_rule(test, BASE, "oracle"), BASE)
        rows["naive"] = metrics(_apply_sell_rule(test, BASE, "naive"), BASE)
        rows["greedy1.3"] = metrics(greedy_sell(test, BASE, 1.3), BASE)
        rows["max_profit"] = metrics(_apply_sell_rule(test, BASE, "max_profit", pol.ev_coeffs), BASE)
        rows["winrate"] = metrics(_apply_sell_rule(test, BASE, "winrate", pol.ev_coeffs), BASE)
        rows["cvar"] = metrics(_apply_sell_rule(test, BASE, "cvar", pol.ev_coeffs,
                                                risk_offset=pol.cvar_risk_offset), BASE)

        pname = PATTERN_NAMES[prev] if prev is not None else "unknown"
        print(f"\n-- 上週波型 = {prev} ({pname}) --   買入資金比例 f: " +
              " ".join(f"{o[:4]}={pol.buy_fraction[o]:.2f}" for o in OBJECTIVES))
        print(f"  {'strategy':<12}{'mean%':>8}{'median%':>9}{'win%':>7}{'worst%':>8}{'cvar5%':>8}")
        for name in ("oracle", "max_profit", "cvar", "winrate", "greedy1.3", "naive"):
            m = rows[name]
            print(f"  {name:<12}{m['mean%']:>8.1f}{m['median%']:>9.1f}{m['win%']:>7.1f}"
                  f"{m['worst%']:>8.1f}{m['cvar5%']:>8.1f}")


def check_calibration(rng):
    print("\n=== 3. forecaster 波型機率校準 (預測機率 vs 實際命中, 真實週, 觀測前2格) ===")
    buckets = {i: [0, 0] for i in range(10)}  # 10 個機率區間: [預測機率和, 命中數, 計數]
    cnt = {i: 0 for i in range(10)}
    psum = {i: 0.0 for i in range(10)}
    hit = {i: 0 for i in range(10)}
    N = 4000
    for _ in range(N):
        prev = rng.integers(0, 4)
        w = None
        for _ in range(60):
            ww = generate_week(int(prev), rng.integers(0, 2**32))
            if ww["base_price"] == BASE:
                w = ww; break
        if w is None:
            continue
        fc = forecast(BASE, w["prices"][:2], prev_pattern=int(prev), n_samples=1)
        if not fc["feasible"]:
            continue
        for k in range(4):
            p = fc["pattern_posterior"][k]
            b = min(9, int(p * 10))
            cnt[b] += 1; psum[b] += p
            if w["pattern"] == k:
                hit[b] += 1
    print(f"  {'pred prob':>12}{'actual':>9}{'count':>8}")
    for b in range(10):
        if cnt[b] == 0:
            continue
        print(f"  {psum[b]/cnt[b]*100:>11.1f}%{hit[b]/cnt[b]*100:>8.1f}%{cnt[b]:>8}")


if __name__ == "__main__":
    rng = np.random.default_rng(2020)
    t0 = time.time()
    check_sampler_fidelity(rng)
    compare_policies(rng)
    check_calibration(rng)
    print(f"\n總耗時 {time.time()-t0:.0f}s")
