#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webapp/advisor_api.py — 網頁 UI 的薄包裝層 (前端唯一會呼叫的後端介面)

職責 (核心演算法都在 core/, 這裡只做「網頁膠水」):
  1. 載入 data/policies.pkl 的預訓練策略 (precompute.py 產出), 免現場訓練;
     查無快取時才退回 train_policy 現場訓練。
  2. 把 Advisor 的「資金比例 f / 賣出比例」換算成「顆數 / 鈴錢」, 並強制顆數為 10 的倍數
     (買入無條件捨去到 10; 賣出比例 × 目前持有後捨去到 10, 全賣則賣光持有)。
  3. 追蹤使用者部位由前端負責 (傳入 holding); 本層只做純函式換算, 不存狀態。

對外只暴露 get_advisor() 與 advise()。回傳皆為可 JSON 化的純 dict / list。
"""

from __future__ import annotations

import math
import os
import pickle
import sys

import numpy as np

# --- 讓 core/ 的扁平模組可被 import (turnip_sim/forecaster/policy_dp/recommend) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORE = os.path.join(_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from forecaster import PATTERN_NAMES, DAY_LABELS, N_SLOTS, forecast  # noqa: E402
from policy_dp import train_policy  # noqa: E402
from recommend import Advisor  # noqa: E402

POLICIES_PKL = os.path.join(_ROOT, "data", "policies.pkl")
BUY_PRICE_RANGE = range(90, 111)          # 官方週日買價 90~110 (閉區間)
PREV_PATTERNS = (None, 0, 1, 2, 3)        # None = 上週波型不知道
QTY_STEP = 10                             # 顆數必為 10 的倍數

# 4 種策略 (保守→積極): 保本勝率 → 穩健保守 → Kelly 成長 → 全力衝刺。
STRATEGIES = ("winrate", "cvar", "kelly", "max_profit")
STRATEGY_INFO = {
    "winrate":    {"label": "保本勝率", "desc": "最保守：一有獲利就賣出，盡量場場不賠"},
    "cvar":       {"label": "穩健保守", "desc": "保守：壓低最慘 5% 的虧損，下注較小"},
    "kelly":      {"label": "Kelly 成長", "desc": "均衡：長期複利成長最佳的下注比例"},
    "max_profit": {"label": "全力衝刺", "desc": "積極：追求期望獲利最大，傾向全押"},
}

# 波型代碼 → 中文 (統一用 三期型/四期型, 不用大爆衝/小爆衝)。
PATTERN_ZH = {0: "波動型", 1: "三期型", 2: "遞減型", 3: "四期型"}

# 上週波型按鈕 → 後端 prev_pattern 代碼。(三期=large_spike=1, 四期=small_spike=3)
PREV_PATTERN_CHOICES = (
    ("不知道", None),
    ("波動型", 0),
    ("四期型", 3),
    ("三期型", 1),
    ("遞減型", 2),
)


def round_down_10(x) -> int:
    """無條件捨去到 10 的倍數 (顆數規則)。"""
    return int(x) // QTY_STEP * QTY_STEP


def _fill_forward(seq, fallback):
    """把 None 以前一個已知值補上 (供 decide_sell 的特徵計算); 開頭未知用 fallback。"""
    out, last = [], fallback
    for v in seq:
        if v is not None:
            last = v
        out.append(last)
    return out


# --------------------------------------------------------------------------
# 載入 / 建立 Advisor (用預訓練快取, 查無才現場訓練)
# --------------------------------------------------------------------------
class CachedAdvisor(Advisor):
    """以 data/policies.pkl 的預訓練 Policy 取代現場 train_policy。"""

    def __init__(self, policies, base_qty=10000):
        super().__init__(base_qty=base_qty)
        self._cache = dict(policies)   # (買價, 上週波型) -> Policy

    def policy(self, buy_price, prev_pattern):
        key = (int(buy_price), prev_pattern)
        if key not in self._cache:     # 快取未覆蓋 (理論上不會) -> 現場補訓
            self._cache[key] = train_policy(int(buy_price), prev_pattern)
        return self._cache[key]


def get_advisor(base_qty=10000) -> CachedAdvisor:
    """載入預訓練快取建立 Advisor。找不到 pkl 時回傳空快取 (改現場訓練)。"""
    policies = {}
    if os.path.exists(POLICIES_PKL):
        with open(POLICIES_PKL, "rb") as f:
            policies = pickle.load(f)
    return CachedAdvisor(policies, base_qty=base_qty)


# --------------------------------------------------------------------------
# 主入口: 給前端的單次建議 (含顆數換算與 10 倍數取整)
# --------------------------------------------------------------------------
def advise(advisor, buy_price, prev_pattern, observed, budget, holding=0):
    """
    參數:
      advisor     : get_advisor() 取得的實例
      buy_price   : 週日買價 (int)
      prev_pattern: None/0/1/2/3
      observed    : 已輸入的逐期賣價 list (None 代表該期略過/未輸入; 末端可省略)
      budget      : 可用鈴錢 (用於買入顆數換算)
      holding     : 目前持有顆數 (用於賣出顆數換算; 必為 10 倍數)

    回傳純 dict (見下方鍵)。
    """
    buy_price = int(buy_price)
    observed = list(observed)
    cur_slot = (len(observed) - 1) if observed else None
    # 容錯模式: 與規則矛盾時改取最接近的子波型 (tolerance_used = 容錯偏差鈴)
    fc = forecast(buy_price, observed, prev_pattern=prev_pattern, auto_tolerance=True)

    out = {
        "buy_price": buy_price,
        "prev_pattern": prev_pattern,
        "feasible": fc.get("feasible", False),
        "tolerance_used": int(fc.get("tolerance_used", 0) or 0),
        "pattern_posterior": {},
        "table": [],
        "remaining_max": None,
        "buy": {},
        "sell": None,
        "current_slot": cur_slot,
    }

    if not fc.get("feasible", False):
        out["error"] = "輸入的價格與官方規則矛盾，請檢查是否輸入錯誤"
        return out

    # 波型後驗機率 (中文名)
    out["pattern_posterior"] = {
        PATTERN_NAMES[k]: float(fc["pattern_posterior"][k]) for k in range(4)
    }

    # 12 時段預測表
    for i in range(N_SLOTS):
        o = fc["observed"][i]
        out["table"].append({
            "slot": i,
            "label": DAY_LABELS[i],
            "obs": (int(o) if o is not None else None),
            "gmin": int(fc["guaranteed_min"][i]),
            "q10": int(round(fc["q10"][i])),
            "q50": int(round(fc["q50"][i])),
            "q90": int(round(fc["q90"][i])),
            "smax": int(fc["slot_max"][i]),
        })

    out["remaining_max"] = {
        "mean": int(round(fc["remaining_max_mean"])),
        "q10": int(round(fc["remaining_max_q10"])),
        "q50": int(round(fc["remaining_max_q50"])),
    }

    pol = advisor.policy(buy_price, prev_pattern)

    # ---- 週日買入建議: 資金比例 → 顆數 (捨去到 10) ----
    buyrec = pol.recommend_buy(capital=budget, base_qty=advisor.base_qty)
    for s in STRATEGIES:
        b = buyrec[s]
        f = b["capital_fraction"]
        spend_budget = budget * f
        qty = round_down_10(spend_budget // buy_price) if f > 0 else 0
        bells = qty * buy_price
        out["buy"][s] = {
            "fraction": f,
            "all_in": b["all_in"],
            "qty": qty,
            "bells": bells,
            "tag": "ALL-IN 全押" if b["all_in"] else (f"{f*100:.0f}% 資金" if f > 0 else "不買"),
        }

    # ---- 本期賣出建議: 比例 × 目前持有 → 顆數 (捨去到 10; 全賣=賣光) ----
    if cur_slot is not None:
        cur_p = fc["observed"][cur_slot]
        cur_price = int(cur_p) if cur_p is not None else 0
        # 續抱價值用 forecaster 的條件化剩餘最高價期望 (知道尖峰還沒到), 不用粗略回歸
        cont = fc.get("remaining_max_mean")
        sdec = pol.decide_sell(_fill_forward(observed, buy_price), cur_slot, cont=cont)
        sell = {"slot": cur_slot, "price": cur_price}
        for s in STRATEGIES:
            frac = sdec[s]
            if frac >= 0.999:
                qty = int(holding)              # 全賣 (holding 已是 10 倍數)
                act = "全部賣出"
            elif frac <= 1e-6:
                qty = 0
                act = "不賣 (續抱)"
            else:
                qty = round_down_10(holding * frac)
                act = f"賣出 {frac*100:.0f}%"
            sell[s] = {"fraction": frac, "qty": qty, "bells": qty * cur_price, "action": act}
        out["sell"] = sell

    return out


def slot_sell_qty(advisor, buy_price, prev_pattern, prices, slot, holding, strategy):
    """單一 slot 的建議賣出顆數。續抱價值用 forecaster 的條件化剩餘最高價期望 (準, 知道尖峰)。
    回傳 (顆數, 文字說明)。holding 為該 slot 當下的持有顆數 (10 倍數)。"""
    if prices[slot] is None or holding <= 0:
        return 0, "—"
    pol = advisor.policy(int(buy_price), prev_pattern)
    obs_prefix = [p for p in prices[:slot + 1]]
    fc = forecast(int(buy_price), obs_prefix, prev_pattern=prev_pattern, auto_tolerance=True)
    cont = fc.get("remaining_max_mean") if fc.get("feasible") else None
    filled = _fill_forward(prices[:slot + 1], int(buy_price))
    frac = pol.decide_sell(filled, slot, cont=cont)[strategy]
    if frac >= 0.999:
        return int(holding), "全部賣出"
    if frac <= 1e-6:
        return 0, "不賣 (續抱)"
    return round_down_10(holding * frac), f"賣出 {frac*100:.0f}%"


def pattern_table(buy_price, prev_pattern, observed, auto_tolerance=True):
    """ac-turnip 風格表格資料: 每個波型 (列) × 12 時段 (欄) 的價格區間 + 後驗機率。
    用解析邊界 (不取樣, 便宜)。auto_tolerance=True 時, 與規則矛盾改取最接近波型。回傳:
      {feasible, rows:[{pattern,name,prob,cells}], overall, obs,
       tolerance_used, violations:[{slot,obs,lo,hi,diff}]}
    rows 依 波動型→三期型→遞減型→四期型, 只含目前仍可行 (機率>0) 的波型。
    """
    from forecaster import enumerate_subpatterns, _select_feasible, N_SLOTS as _NS

    base = int(buy_price)
    obs = list(observed) + [None] * (_NS - len(observed))
    obs = obs[:_NS]
    obs_idx = [i for i, v in enumerate(obs) if v is not None]

    from forecaster import UNKNOWN_PRIOR, TRANSITION

    subs = enumerate_subpatterns(base, prev_pattern)
    feas, tol_used = _select_feasible(subs, obs, obs_idx, auto_tolerance)

    if not feas:
        return {"feasible": False, "tolerance_needed": tol_used}

    # 機率對齊 mikebryant: 權重 = 子波型先驗 (波型轉移 × 結構均勻), 在所有可行子波型間正規化
    # -> 波型機率 ∝ 先驗 × 可行設定占比 (大部分結構被排除的波型, 機率按比例下降)
    ws = [pr * math.exp(ll - max(t[2] for t in feas)) for (_, pr, ll, _) in feas]
    tot = sum(ws) or 1.0
    ws = [w / tot for w in ws]

    INF = float("inf")
    overall_min = [INF] * _NS
    overall_max = [-INF] * _NS
    rows = []
    for pat in (0, 1, 2, 3):
        idxs = [k for k, (sp, _, _, _) in enumerate(feas) if sp.pattern == pat]
        if not idxs:
            continue
        prob = sum(ws[k] for k in idxs)
        if prob < 0.005:        # 機率趨近 0 的波型不列出
            continue
        cmin = [INF] * _NS
        cmax = [-INF] * _NS
        for k in idxs:
            cond = feas[k][3]                    # 條件化邊界 (已沿倍率鏈傳遞觀測, 已釘觀測格)
            for i in range(_NS):
                cmin[i] = min(cmin[i], cond[i][0])
                cmax[i] = max(cmax[i], cond[i][1])
        for i in range(_NS):
            overall_min[i] = min(overall_min[i], cmin[i])
            overall_max[i] = max(overall_max[i], cmax[i])
        rows.append({"pattern": pat, "name": PATTERN_ZH[pat], "prob": prob,
                     "cells": list(zip(cmin, cmax))})

    overall = list(zip(overall_min, overall_max))
    return {"feasible": True, "rows": rows, "overall": overall, "obs": obs,
            "tolerance_used": int(tol_used), "violations": []}


def pattern_bands(buy_price, prev_pattern, observed, pct=0.9, n=3000, seed=0):
    """每個波型 (含『所有波型』) 的『最可能價格』中央區間 (涵蓋 pct 機率) per slot。
    用子波型取樣 (依先驗×似然加權), 回傳 {名稱: {lo:[12], hi:[12]}}。
    pct=0.5/0.7/0.9 -> 取中央 [ (1-pct)/2 , 1-(1-pct)/2 ] 分位數。"""
    from forecaster import (enumerate_subpatterns, _select_feasible,
                            _weighted_quantile, N_SLOTS as _NS)
    base = int(buy_price)
    obs = list(observed) + [None] * (_NS - len(observed))
    obs = obs[:_NS]
    obs_idx = [i for i, v in enumerate(obs) if v is not None]
    feas, _ = _select_feasible(enumerate_subpatterns(base, prev_pattern),
                              obs, obs_idx, auto_tolerance=True)
    if not feas:
        return {}
    rng = np.random.default_rng(seed)
    mx = max(t[2] for t in feas)
    sampled = [(sp, sp.prior * math.exp(ll - mx), sp.sample(base, n, rng))
               for (sp, _, ll, _) in feas]
    qlo, qhi = (1 - pct) / 2, 1 - (1 - pct) / 2

    def band(items):
        tot = sum(w for _, w, _ in items) or 1.0
        S = np.concatenate([s for _, _, s in items], axis=0).astype(float)
        W = np.concatenate([np.full(n, w / tot / n) for _, w, _ in items])
        for i in obs_idx:
            S[:, i] = obs[i]
        lo = [int(round(_weighted_quantile(S[:, i], W, qlo))) for i in range(_NS)]
        hi = [int(round(_weighted_quantile(S[:, i], W, qhi))) for i in range(_NS)]
        return {"lo": lo, "hi": hi}

    out = {"所有波型": band(sampled)}
    for pat in (0, 1, 2, 3):
        items = [t for t in sampled if t[0].pattern == pat]
        if items:
            out[PATTERN_ZH[pat]] = band(items)
    return out


# --------------------------------------------------------------------------
# smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    a = get_advisor()
    print("快取策略數:", len(a._cache), "(0 表示尚未 precompute, 將現場訓練)")
    r = advise(a, 100, prev_pattern=2, observed=[88, 84], budget=1_000_000, holding=10000)
    print("feasible:", r["feasible"])
    print("波型後驗:", {k: round(v, 3) for k, v in r["pattern_posterior"].items()})
    print("買入(各策略):", {k: (v["qty"], v["tag"]) for k, v in r["buy"].items()})
    print("賣出(各策略):", None if r["sell"] is None
          else {k: (r["sell"][k]["qty"], r["sell"][k]["action"]) for k in STRATEGIES})
    print("表格首兩列:", r["table"][:2])
