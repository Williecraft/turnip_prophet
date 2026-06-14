#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecaster.py — 大頭菜「機率預測引擎」(belief engine)

給定本週買價 (Daisy Mae 週日價)、可選的上週波型、以及目前為止觀測到的時段價格,
輸出對「本週剩餘走勢」的機率預測, 供 policy_dp.py 做買賣決策。

輸出:
  - pattern_posterior : 四種波型的後驗機率 P(pattern | 觀測)
  - per-slot 價格分布 : guaranteed-min / max / mean / 分位數 / P(price>=x)
  - remaining_max     : 「本週剩餘最高價」的分布 (賣出決策最關鍵的量)

做法 (列舉子波型 enumerate sub-patterns, 類 Turnip Prophet):
  1. 每種波型的離散結構有限可枚舉 (peakStart / 相位長度), 每個「子波型」給先驗權重
     = 上週波型轉移機率 x 結構參數均勻機率。prev_pattern=None 時用 stationary 邊際分布。
  2. 用觀測價格篩掉不可行子波型 (任一時段超出該子波型 [min,max] 即排除),
     並以 1/區間寬 當離散均勻 likelihood 修正 -> 後驗權重。
  3. 每個可行子波型 forward 取樣 (numpy 向量化) 還原完整價格路徑 (週內相關性正確);
     未來「尖峰」時段是獨立抽樣, forward sample 對其為精確。
     近似: 跨觀測邊界、仍在進行中的「遞減段」幅度未逐格條件化 (低價值區段, 影響小),
           校準度於 backtest.py 檢驗。

價格分布只用於『策略』, 不需 bit-exact 種子: 官方只用均勻抽樣, 故以標準均勻 RNG 取樣
在統計上與官方一致 (第一階段已驗證 peak 倍率/轉移機率吻合)。guaranteed-min 用解析邊界
(對可行子波型取 min), 為精確值。
"""

from __future__ import annotations

import numpy as np

from turnip_sim import PATTERN_NAMES, DAY_LABELS

N_SLOTS = 12  # 週一上午 ~ 週六下午

# ---- 官方 pattern 轉移機率 P[prev][next] (由 ALGORITHM.md 門檻表推得) ----
TRANSITION = np.array([
    [0.20, 0.30, 0.15, 0.35],   # prev 0 fluctuating
    [0.50, 0.05, 0.20, 0.25],   # prev 1 large spike
    [0.25, 0.45, 0.05, 0.25],   # prev 2 decreasing
    [0.45, 0.25, 0.15, 0.15],   # prev 3 small spike
])


def stationary_distribution() -> np.ndarray:
    """轉移矩陣的長期(stationary)邊際分布; prev_pattern 未知時當先驗。"""
    vals, vecs = np.linalg.eig(TRANSITION.T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, i])
    return pi / pi.sum()


STATIONARY = stationary_distribution()


def _intceil(x):
    """官方 intceil 的向量化版 (正價格時等同向上取整)。"""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.99999).astype(np.int64)


# =============================================================================
# 子波型: 每個提供 (1) forward 取樣 sample(base, n, rng)->(n,12) int
#                    (2) 解析邊界 bounds(base)->(12,2) int  [min,max] 每時段
#         以及 pattern 類型與先驗權重 prior。
# 全部以「work 槽 2..13」對齊官方; 對外輸出 slot = work-2 (0..11)。
# =============================================================================

def _dec_run_forward(base, n, rng, start_lo, start_hi, fixed_dec, rand_dec, length):
    """遞減段 forward 取樣, 回傳 (n, length) 價格與「離開時的 rate」(供接續)。"""
    rate = rng.uniform(start_lo, start_hi, size=n)
    out = np.empty((n, length), dtype=np.int64)
    for i in range(length):
        out[:, i] = _intceil(rate * base)
        rate = rate - fixed_dec - rng.uniform(0.0, rand_dec, size=n)
    return out, rate


def _dec_run_bounds(base, start_lo, start_hi, fixed_dec, rand_dec, length):
    """遞減段每格 [min,max]: max=start_hi - i*fixed_dec; min=start_lo - i*(fixed_dec+rand_dec)。"""
    b = np.empty((length, 2), dtype=np.int64)
    for i in range(length):
        hi_rate = start_hi - i * fixed_dec
        lo_rate = start_lo - i * (fixed_dec + rand_dec)
        b[i, 0] = _intceil(lo_rate * base)
        b[i, 1] = _intceil(hi_rate * base)
    return b


def _indep_bounds(base, a, b):
    return int(_intceil(a * base)), int(_intceil(b * base))


class SubPattern:
    __slots__ = ("pattern", "prior", "_sampler", "_bounds")

    def __init__(self, pattern, prior, sampler, bounds):
        self.pattern = pattern
        self.prior = prior
        self._sampler = sampler      # fn(base, n, rng) -> (n,12)
        self._bounds = bounds        # (12,2)

    def sample(self, base, n, rng):
        return self._sampler(base, n, rng)

    def bounds(self):
        return self._bounds


# ---- Pattern 2: Decreasing (單一結構) ----
def _make_p2(base, prior):
    bounds = _dec_run_bounds(base, 0.85, 0.90, 0.03, 0.02, 12)

    def sampler(base, n, rng):
        out, _ = _dec_run_forward(base, n, rng, 0.85, 0.90, 0.03, 0.02, 12)
        return out

    return SubPattern(2, prior, sampler, bounds)


# ---- Pattern 1: Large spike, peakStart(work) in 3..9 ----
def _make_p1(base, prior, peak_start):
    dec_len = peak_start - 2
    spike_factors = [(0.9, 1.4), (1.4, 2.0), (2.0, 6.0), (1.4, 2.0), (0.9, 1.4)]
    rest_start_slot = (peak_start - 2) + dec_len + 5  # = peak_start+3 work -> slot
    # build bounds
    bounds = np.empty((12, 2), dtype=np.int64)
    bounds[:dec_len] = _dec_run_bounds(base, 0.85, 0.90, 0.03, 0.02, dec_len)
    for k, (a, b) in enumerate(spike_factors):
        bounds[dec_len + k] = _indep_bounds(base, a, b)
    for s in range(dec_len + 5, 12):
        bounds[s] = _indep_bounds(base, 0.4, 0.9)

    def sampler(base, n, rng):
        out = np.empty((n, 12), dtype=np.int64)
        dec, _ = _dec_run_forward(base, n, rng, 0.85, 0.90, 0.03, 0.02, dec_len)
        out[:, :dec_len] = dec
        for k, (a, b) in enumerate(spike_factors):
            out[:, dec_len + k] = _intceil(rng.uniform(a, b, size=n) * base)
        for s in range(dec_len + 5, 12):
            out[:, s] = _intceil(rng.uniform(0.4, 0.9, size=n) * base)
        return out

    return SubPattern(1, prior, sampler, bounds)


# ---- Pattern 3: Small spike, peakStart(work) in 2..9 ----
def _make_p3(base, prior, peak_start):
    dec_len = peak_start - 2  # 可為 0
    # spike 五格: hi, hi, side(-1), peak, side(-1) ; side/peak 共用 rate2~U(1.4,2.0)
    bounds = np.empty((12, 2), dtype=np.int64)
    if dec_len > 0:
        bounds[:dec_len] = _dec_run_bounds(base, 0.4, 0.9, 0.03, 0.02, dec_len)
    s0 = dec_len
    bounds[s0] = _indep_bounds(base, 0.9, 1.4)
    bounds[s0 + 1] = _indep_bounds(base, 0.9, 1.4)
    bounds[s0 + 2] = (int(_intceil(1.4 * base)) - 1, int(_intceil(2.0 * base)) - 1)
    bounds[s0 + 3] = (int(_intceil(1.4 * base)), int(_intceil(2.0 * base)))
    bounds[s0 + 4] = (int(_intceil(1.4 * base)) - 1, int(_intceil(2.0 * base)) - 1)
    rest_from = s0 + 5
    if rest_from < 12:
        bounds[rest_from:] = _dec_run_bounds(base, 0.4, 0.9, 0.03, 0.02, 12 - rest_from)

    def sampler(base, n, rng):
        out = np.empty((n, 12), dtype=np.int64)
        if dec_len > 0:
            dec, _ = _dec_run_forward(base, n, rng, 0.4, 0.9, 0.03, 0.02, dec_len)
            out[:, :dec_len] = dec
        out[:, s0] = _intceil(rng.uniform(0.9, 1.4, size=n) * base)
        out[:, s0 + 1] = _intceil(rng.uniform(0.9, 1.4, size=n) * base)
        rate2 = rng.uniform(1.4, 2.0, size=n)
        out[:, s0 + 2] = _intceil(rng.uniform(1.4, rate2) * base) - 1
        out[:, s0 + 3] = _intceil(rate2 * base)
        out[:, s0 + 4] = _intceil(rng.uniform(1.4, rate2) * base) - 1
        if rest_from < 12:
            dec2, _ = _dec_run_forward(base, n, rng, 0.4, 0.9, 0.03, 0.02, 12 - rest_from)
            out[:, rest_from:] = dec2
        return out

    return SubPattern(3, prior, sampler, bounds)


# ---- Pattern 0: Fluctuating ----
def _make_p0(base, prior, dec_len1, hi1, hi3):
    dec_len2 = 5 - dec_len1
    hi2 = (7 - hi1) - hi3
    # segment order: hi1*hi, dec_len1*dec, hi2*hi, dec_len2*dec, hi3*hi
    seg = []
    seg += [("hi",)] * hi1
    seg += [("d1",)] * dec_len1
    seg += [("hi",)] * hi2
    seg += [("d2",)] * dec_len2
    seg += [("hi",)] * hi3
    assert len(seg) == 12

    # bounds
    bounds = np.empty((12, 2), dtype=np.int64)
    d1b = _dec_run_bounds(base, 0.6, 0.8, 0.04, 0.06, dec_len1)
    d2b = _dec_run_bounds(base, 0.6, 0.8, 0.04, 0.06, dec_len2)
    hib = _indep_bounds(base, 0.9, 1.4)
    di1 = di2 = 0
    for i, s in enumerate(seg):
        if s[0] == "hi":
            bounds[i] = hib
        elif s[0] == "d1":
            bounds[i] = d1b[di1]; di1 += 1
        else:
            bounds[i] = d2b[di2]; di2 += 1

    def sampler(base, n, rng):
        out = np.empty((n, 12), dtype=np.int64)
        d1, _ = _dec_run_forward(base, n, rng, 0.6, 0.8, 0.04, 0.06, dec_len1)
        d2, _ = _dec_run_forward(base, n, rng, 0.6, 0.8, 0.04, 0.06, dec_len2)
        di1 = di2 = 0
        for i, s in enumerate(seg):
            if s[0] == "hi":
                out[:, i] = _intceil(rng.uniform(0.9, 1.4, size=n) * base)
            elif s[0] == "d1":
                out[:, i] = d1[:, di1]; di1 += 1
            else:
                out[:, i] = d2[:, di2]; di2 += 1
        return out

    return SubPattern(0, prior, sampler, bounds)


def enumerate_subpatterns(base, prev_pattern):
    """列舉所有子波型並給先驗權重 (含上週波型轉移 x 結構均勻)。"""
    if prev_pattern is None:
        pat_prior = STATIONARY
    else:
        pat_prior = TRANSITION[prev_pattern]

    subs = []
    # pattern 0: dec_len1 in {2,3}, hi1 in 0..6, hi3 in 0..(6-hi1)
    for dec_len1 in (2, 3):
        for hi1 in range(7):
            for hi3 in range(0, 7 - hi1):
                w = pat_prior[0] * 0.5 * (1 / 7) * (1 / (7 - hi1))
                subs.append(_make_p0(base, w, dec_len1, hi1, hi3))
    # pattern 1: peakStart 3..9
    for ps in range(3, 10):
        subs.append(_make_p1(base, pat_prior[1] * (1 / 7), ps))
    # pattern 2
    subs.append(_make_p2(base, pat_prior[2]))
    # pattern 3: peakStart 2..9
    for ps in range(2, 10):
        subs.append(_make_p3(base, pat_prior[3] * (1 / 8), ps))
    return subs


def feasible_bounds(base, observed, prev_pattern=None):
    """輕量版 (不取樣): 只回傳 (guaranteed_min[12], slot_max[12]); 觀測矛盾回 None。
    解析邊界工具 (不需取樣即可取得每時段保證下限/上限)。"""
    obs = list(observed) + [None] * (N_SLOTS - len(observed))
    obs = obs[:N_SLOTS]
    obs_idx = [i for i, v in enumerate(obs) if v is not None]
    gmin = np.full(N_SLOTS, np.inf)
    smax = np.full(N_SLOTS, -np.inf)
    any_feas = False
    for sp in enumerate_subpatterns(int(base), prev_pattern):
        b = sp.bounds()
        if all(b[i, 0] <= obs[i] <= b[i, 1] for i in obs_idx):
            any_feas = True
            gmin = np.minimum(gmin, b[:, 0])
            smax = np.maximum(smax, b[:, 1])
    if not any_feas:
        return None
    return gmin.astype(int), smax.astype(int)


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(q, cw, v)


def forecast(buy_price, observed, prev_pattern=None, n_samples=4000, seed=0):
    """主入口。
    observed: 長度<=12 的序列, 每元素為已觀測整數價格或 None(未知); 缺尾自動補 None。
    回傳 dict (見模組說明)。
    """
    base = int(buy_price)
    obs = list(observed) + [None] * (N_SLOTS - len(observed))
    obs = obs[:N_SLOTS]
    obs_idx = [i for i, v in enumerate(obs) if v is not None]

    rng = np.random.default_rng(seed)
    subs = enumerate_subpatterns(base, prev_pattern)

    feas = []
    for sp in subs:
        b = sp.bounds()
        ok = True
        loglik = 0.0
        for i in obs_idx:
            lo, hi = int(b[i, 0]), int(b[i, 1])
            if not (lo <= obs[i] <= hi):
                ok = False
                break
            loglik += -np.log(hi - lo + 1)   # 離散均勻 likelihood 1/區間寬
        if ok:
            feas.append((sp, sp.prior, loglik))

    if not feas:
        # 觀測與任何官方子波型矛盾 (輸入錯誤或首週特例) -> 回傳空信念旗標
        return {"feasible": False, "buy_price": base, "observed": obs,
                "pattern_posterior": {k: 0.0 for k in range(4)}}

    # 後驗權重 = prior * likelihood
    max_ll = max(t[2] for t in feas)
    weights = np.array([prior * np.exp(ll - max_ll) for (_, prior, ll) in feas])
    weights = weights / weights.sum()

    # forward 取樣彙整
    all_samples = []
    all_w = []
    pat_post = np.zeros(4)
    guaranteed_min = np.full(N_SLOTS, np.inf)
    slot_max = np.full(N_SLOTS, -np.inf)
    for (sp, _, _), w in zip(feas, weights):
        pat_post[sp.pattern] += w
        b = sp.bounds()
        guaranteed_min = np.minimum(guaranteed_min, b[:, 0])
        slot_max = np.maximum(slot_max, b[:, 1])
        s = sp.sample(base, n_samples, rng)
        all_samples.append(s)
        all_w.append(np.full(n_samples, w / n_samples))

    samples = np.concatenate(all_samples, axis=0)        # (M,12)
    sw = np.concatenate(all_w)                            # (M,)

    # 把已觀測時段「釘死」成觀測值 (forward 樣本在這些格不一定等於觀測)
    for i in obs_idx:
        samples[:, i] = obs[i]

    # per-slot 統計 (只對未來/未觀測時段有意義, 已觀測格為定值)
    mean = np.average(samples, axis=0, weights=sw)
    q10 = np.array([_weighted_quantile(samples[:, i], sw, 0.10) for i in range(N_SLOTS)])
    q50 = np.array([_weighted_quantile(samples[:, i], sw, 0.50) for i in range(N_SLOTS)])
    q90 = np.array([_weighted_quantile(samples[:, i], sw, 0.90) for i in range(N_SLOTS)])

    # 剩餘最高價分布 (未觀測時段的 max); 若全部觀測完則為 0 長度
    future_idx = [i for i in range(N_SLOTS) if obs[i] is None]
    if future_idx:
        rem_max = samples[:, future_idx].max(axis=1)
    else:
        rem_max = np.zeros(samples.shape[0], dtype=np.int64)

    return {
        "feasible": True,
        "buy_price": base,
        "prev_pattern": prev_pattern,
        "observed": obs,
        "n_feasible_subpatterns": len(feas),
        "pattern_posterior": {k: float(pat_post[k]) for k in range(4)},
        "guaranteed_min": guaranteed_min.astype(int),   # 每時段保證至少會看到的下限
        "slot_max": slot_max.astype(int),
        "mean": mean,
        "q10": q10, "q50": q50, "q90": q90,
        "future_idx": future_idx,
        "remaining_max_mean": float(np.average(rem_max, weights=sw)),
        "remaining_max_q10": float(_weighted_quantile(rem_max.astype(float), sw, 0.10)),
        "remaining_max_q50": float(_weighted_quantile(rem_max.astype(float), sw, 0.50)),
        "remaining_max_samples": rem_max,
        "remaining_max_weights": sw,
    }


def print_forecast(fc):
    if not fc["feasible"]:
        print("⚠ 觀測與官方規則矛盾 (請檢查輸入或是否為首週特例)")
        return
    print(f"買價 base={fc['buy_price']}  上週波型={fc['prev_pattern']}  "
          f"可行子波型={fc['n_feasible_subpatterns']}")
    print("波型後驗機率:")
    for k in range(4):
        print(f"  {k} {PATTERN_NAMES[k]:<12} {fc['pattern_posterior'][k]*100:5.1f}%")
    print(f"{'slot':<8}{'obs':>6}{'gMin':>7}{'q10':>7}{'q50':>7}{'q90':>7}{'max':>7}")
    for i in range(N_SLOTS):
        o = fc["observed"][i]
        os_ = str(o) if o is not None else "-"
        print(f"{DAY_LABELS[i]:<8}{os_:>6}{fc['guaranteed_min'][i]:>7}"
              f"{fc['q10'][i]:>7.0f}{fc['q50'][i]:>7.0f}{fc['q90'][i]:>7.0f}{fc['slot_max'][i]:>7}")
    print(f"剩餘最高價: 期望={fc['remaining_max_mean']:.0f}  "
          f"q10={fc['remaining_max_q10']:.0f}  q50={fc['remaining_max_q50']:.0f}")


if __name__ == "__main__":
    print("stationary 先驗:", {PATTERN_NAMES[k]: round(float(STATIONARY[k]), 4) for k in range(4)})
    print("\n=== 範例: 買價100, 上週=decreasing(2), 觀測 Mon_AM=88, Mon_PM=84 ===")
    fc = forecast(100, [88, 84], prev_pattern=2)
    print_forecast(fc)
