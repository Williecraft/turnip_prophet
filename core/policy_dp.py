#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_dp.py — 大頭菜 買賣決策引擎 (DP / 最佳停止)

對同一個 (買價, 上週波型) 訓練 4 套策略, 每套輸出「每期賣多少」與「週日買多少(資金比例)」:

  max_profit  最高獲利 (風險中性): EV 最佳停止 (Longstaff–Schwartz 回歸); 收益對持有量線性
              -> 賣出為全賣/全抱門檻; 買入 EV>0 即 all-in。
  kelly       成長最佳 (full Kelly): 賣出同 max_profit(EV 停止), 但買入用 full Kelly 部位
              (最大化長期幾何成長); 介於 max_profit(all-in) 與 cvar(half-Kelly) 之間。
  winrate     ①勝率 (賠最少): 第一次出現 price>=成本即全賣鎖定一次「贏」; 否則 EV 停損。
  cvar        ②尾部風險 (CVaR 5%): 門檻參數在情境上最佳化, 控制最慘 5% 的結果; 買入用
              分數 Kelly/CVaR 決定資金比例。

決策變數: 賣 = 賣出比例; 買 = 資金比例 f (0~1, 1=all-in)。
顆數/鈴錢只是 UI 換算 (顆數=倍數x10000, 鈴錢=顆數x買價), 不進訓練。

情境取樣復用 forecaster 的子波型生成器 (numpy, 與官方統計一致)。
"""

from __future__ import annotations

import numpy as np

from forecaster import enumerate_subpatterns, N_SLOTS

OBJECTIVES = ("max_profit", "kelly", "cvar", "winrate")
CVAR_ALPHA = 0.05        # 最慘 5%
CVAR_RISK_WEIGHT = 0.5   # mean-CVaR 折衷權重 w: 0=純EV(=max_profit), 1=純CVaR(≈不賭); 越大越保守


# --------------------------------------------------------------------------
# 情境取樣 (training/evaluation 用; 與官方分布一致)
# --------------------------------------------------------------------------
def sample_scenarios(base, prev_pattern, n, rng):
    """回傳 (prices[n,12] int, pattern[n] int)。依子波型先驗抽樣。"""
    subs = enumerate_subpatterns(int(base), prev_pattern)
    priors = np.array([sp.prior for sp in subs], dtype=np.float64)
    priors /= priors.sum()
    counts = rng.multinomial(n, priors)
    prices = np.empty((n, N_SLOTS), dtype=np.int64)
    pat = np.empty(n, dtype=np.int64)
    pos = 0
    for sp, c in zip(subs, counts):
        if c:
            prices[pos:pos + c] = sp.sample(int(base), int(c), rng)
            pat[pos:pos + c] = sp.pattern
            pos += c
    return prices, pat


# --------------------------------------------------------------------------
# EV 最佳停止 (Longstaff–Schwartz): 學每個時段的「續抱價值」回歸係數
# --------------------------------------------------------------------------
def _features(prices, base, t):
    """slot t 的回歸特徵 (用觀測到 t 為止的資訊)。"""
    r = prices[:, t] / base
    rmax = prices[:, :t + 1].max(axis=1) / base
    return np.column_stack([np.ones(len(r)), r, r * r, rmax, rmax * rmax])


def train_ev_stopping(prices, base):
    """回傳每個 slot 的續抱價值回歸係數 (slot 11 為 None=必賣)。"""
    n, T = prices.shape
    V = prices[:, T - 1].astype(np.float64).copy()   # 撐到最後一格必須賣
    coeffs = [None] * T
    for t in range(T - 2, -1, -1):
        X = _features(prices, base, t)
        c, *_ = np.linalg.lstsq(X, V, rcond=None)
        cont = X @ c
        sell = prices[:, t] >= cont
        V = np.where(sell, prices[:, t].astype(np.float64), V)
        coeffs[t] = c
    return coeffs


def _ev_continuation(observed_prices, base, t, coeffs):
    """runtime: 用觀測價算 slot t 的續抱價值 (單一情境)。"""
    if coeffs[t] is None:
        return -np.inf  # 最後一格 -> 一定賣
    arr = np.array(observed_prices, dtype=np.float64).reshape(1, -1)
    X = _features(arr, base, t)
    return float((X @ coeffs[t]).item())


# --------------------------------------------------------------------------
# 策略容器
# --------------------------------------------------------------------------
class Policy:
    """針對單一 (買價, 上週波型) 訓練好的 4 套策略。"""

    def __init__(self, base, prev_pattern, ev_coeffs, cvar_risk_offset, buy_fraction, stats):
        self.base = base
        self.prev_pattern = prev_pattern
        self.ev_coeffs = ev_coeffs
        self.cvar_risk_offset = cvar_risk_offset  # CVaR: 續抱門檻的調降量 (佔 base, <0)
        self.buy_fraction = buy_fraction          # dict objective -> 資金比例 f
        self.stats = stats                        # dict objective -> 每單位報酬統計

    # ---- 賣出決策: 回傳「現在賣出佔『剩餘持有』的比例」每目標 (全賣=1.0/全抱=0.0) ----
    def decide_sell(self, observed_prices, t, cont=None):
        """cont: 續抱價值 (= 之後仍能拿到的期望最佳價)。預設用 ev 回歸; 但回歸只看
        (現價, 至今最高) 兩個特徵, 無法分辨『暴跌後剛起跳=爆衝前兆』, 會在尖峰前過早賣出。
        故 runtime 由 forecaster 的『條件化剩餘最高價期望』傳入更準的 cont。"""
        price = observed_prices[t]
        base = self.base
        last = (t == N_SLOTS - 1)
        if cont is None:
            cont = _ev_continuation(observed_prices, base, t, self.ev_coeffs)
        out = {}
        out["max_profit"] = 1.0 if (last or price >= cont) else 0.0
        out["kelly"] = out["max_profit"]   # 賣出規則同 max_profit (EV 停止)
        out["winrate"] = 1.0 if (last or price >= base or price >= cont) else 0.0
        out["cvar"] = 1.0 if (last or price >= cont + self.cvar_risk_offset * base) else 0.0
        return out

    # ---- 買入建議: 資金比例 f + (可選資金/基礎量換算) ----
    def recommend_buy(self, capital=None, base_qty=10000):
        recs = {}
        for obj in OBJECTIVES:
            f = self.buy_fraction[obj]
            rec = {"capital_fraction": f, "all_in": f >= 0.999}
            if capital is not None and f > 0:
                bells = int(capital * f)
                qty = bells // self.base
                rec.update(bells=qty * self.base, quantity=qty,
                           multiplier_of_base=qty / base_qty)
            recs[obj] = rec
        return recs


# --------------------------------------------------------------------------
# 在情境上套用某賣出規則 -> 每情境的「每單位最終收入」 (向量化)
# --------------------------------------------------------------------------
def _apply_sell_rule(prices, base, rule, ev_coeffs=None, risk_offset=0.0):
    """rule in {max_profit, winrate, cvar, oracle, naive}. 回傳每情境『每單位收入』U[n] (全賣/全抱)。
    cvar 用『風險調整門檻』= EV 續抱價值 + risk_offset*base (risk_offset<0 -> 較早賣, 砍尾部風險)。"""
    n, T = prices.shape
    if rule == "oracle":
        return prices.max(axis=1).astype(np.float64)
    if rule == "naive":
        return prices[:, -1].astype(np.float64)

    sold = np.zeros(n, dtype=bool)
    U = np.zeros(n, dtype=np.float64)
    for t in range(T):
        last = (t == T - 1)
        price = prices[:, t].astype(np.float64)
        if ev_coeffs is not None and ev_coeffs[t] is not None:
            cont = _features(prices, base, t) @ ev_coeffs[t]
        else:
            cont = np.full(n, -np.inf)

        if rule == "max_profit":
            trigger = price >= cont
        elif rule == "winrate":
            trigger = (price >= base) | (price >= cont)
        elif rule == "cvar":
            trigger = price >= (cont + risk_offset * base)
        else:
            raise ValueError(rule)
        if last:
            trigger = np.ones(n, dtype=bool)

        do = trigger & (~sold)
        U[do] = price[do]
        sold |= do
    return U


def _cvar(values, alpha=CVAR_ALPHA):
    """最慘 alpha 比例的平均 (越大越好)。"""
    k = max(1, int(len(values) * alpha))
    return float(np.sort(values)[:k].mean())


def _mean_cvar_score(values, w=CVAR_RISK_WEIGHT):
    """mean-CVaR 折衷分數: (1-w)*平均 + w*CVaR5。攻守均衡, 不會塌成『不賭』。"""
    return (1 - w) * float(np.mean(values)) + w * _cvar(values)


# --------------------------------------------------------------------------
# 訓練 (對單一 買價+上週波型)
# --------------------------------------------------------------------------
def train_policy(base, prev_pattern, n_train=60000, n_eval=60000, seed=0):
    rng = np.random.default_rng(seed)
    train, _ = sample_scenarios(base, prev_pattern, n_train, rng)
    ev_coeffs = train_ev_stopping(train, base)

    # CVaR: 搜尋 risk_offset (調降續抱門檻 -> 較早賣砍尾部) 最大化『mean-CVaR 折衷分數』
    best_off, best_score = 0.0, -np.inf
    for off in np.round(np.arange(-1.5, 0.001, 0.1), 1):
        U = _apply_sell_rule(train, base, "cvar", ev_coeffs, risk_offset=off)
        s = _mean_cvar_score(U)
        if s > best_score:
            best_score, best_off = s, off

    # 在獨立 eval 情境上估各目標『每單位報酬 R=U/base』分布, 決定買入資金比例
    ev_rng = np.random.default_rng(seed + 1)
    ev_scen, _ = sample_scenarios(base, prev_pattern, n_eval, ev_rng)
    U = {
        "max_profit": _apply_sell_rule(ev_scen, base, "max_profit", ev_coeffs),
        "winrate": _apply_sell_rule(ev_scen, base, "winrate", ev_coeffs),
        "cvar": _apply_sell_rule(ev_scen, base, "cvar", ev_coeffs, risk_offset=best_off),
    }
    U["kelly"] = U["max_profit"]   # kelly 賣出規則同 max_profit (EV 停止), 差別只在買入部位

    buy_fraction, stats = {}, {}
    for obj in OBJECTIVES:
        R = U[obj] / base                       # 每單位報酬倍數
        mean_R = float(R.mean())
        winrate = float((U[obj] >= base).mean())
        worst = float(R.min())
        cvar_R = _cvar(R)
        stats[obj] = {"mean_profit_pct": (mean_R - 1) * 100,
                      "winrate_pct": winrate * 100,
                      "worst_pct": (worst - 1) * 100,
                      "cvar5_pct": (cvar_R - 1) * 100}
        # 買入資金比例
        if obj == "max_profit":
            f = 1.0 if mean_R > 1 else 0.0      # EV>0 -> all-in (風險中性, 接受高變異)
        elif obj == "kelly":
            f = _kelly_fraction(R, frac=1.0)    # full Kelly: 長期幾何成長最佳
        elif obj == "winrate":
            # 單週勝率與買入數量無關, 故不能用勝率決定部位; 改用 1/4-Kelly:
            # 複利下『全押+微薄賠率』會被偶發虧損週吃光, 故取最保守的『有下注』部位。
            f = _kelly_fraction(R, frac=0.25) if winrate >= 0.5 else 0.0
        else:  # cvar: half-Kelly 中間部位
            f = _kelly_fraction(R, frac=0.5)
        buy_fraction[obj] = f

    return Policy(base, prev_pattern, ev_coeffs, best_off, buy_fraction, stats)


def _kelly_fraction(R, frac=0.5, grid=None):
    """分數 Kelly 資金比例: 最大化 E[log(1+f(R-1))] 的 f, 再乘 frac (預設 half-Kelly)。
    這是 CVaR/穩健投資的標準『中間部位』sizing — 不會 all-in 也不會不買。"""
    if grid is None:
        grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    growth = 1.0 + grid[:, None] * (R[None, :] - 1.0)
    val = np.log(np.maximum(1e-9, growth)).mean(axis=1)
    best = grid[int(np.argmax(val))]
    return float(min(1.0, frac * best))


# --------------------------------------------------------------------------
# 自我測試: EV 策略應接近 oracle 且勝過 naive; 各目標行為符合預期
# --------------------------------------------------------------------------
if __name__ == "__main__":
    base, prev = 100, 2
    pol = train_policy(base, prev, n_train=40000, n_eval=80000, seed=0)
    rng = np.random.default_rng(999)
    test, _ = sample_scenarios(base, prev, 200000, rng)

    oracle = _apply_sell_rule(test, base, "oracle").mean()
    naive = _apply_sell_rule(test, base, "naive").mean()
    ev = _apply_sell_rule(test, base, "max_profit", pol.ev_coeffs).mean()
    print(f"買價={base} 上週波型={prev}")
    print(f"  oracle(完美後見) 每單位均收入 = {oracle:.1f}")
    print(f"  naive (週六硬賣)              = {naive:.1f}")
    print(f"  EV 最佳停止                   = {ev:.1f}   "
          f"(達 oracle 的 {ev/oracle*100:.1f}%, 勝過 naive {ev/naive*100:.0f}%)")
    print(f"  CVaR risk_offset* = {pol.cvar_risk_offset}")
    print("\n各目標 (每單位報酬 %, 對成本):")
    print(f"  {'objective':<12}{'mean%':>8}{'winrate%':>10}{'worst%':>9}{'cvar5%':>9}{'buy f':>7}")
    for obj in OBJECTIVES:
        s = pol.stats[obj]
        print(f"  {obj:<12}{s['mean_profit_pct']:>8.1f}{s['winrate_pct']:>10.1f}"
              f"{s['worst_pct']:>9.1f}{s['cvar5_pct']:>9.1f}{pol.buy_fraction[obj]:>7.2f}")
