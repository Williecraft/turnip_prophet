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

import os
import pickle
import sys

# --- 讓 core/ 的扁平模組可被 import (turnip_sim/forecaster/policy_dp/recommend) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORE = os.path.join(_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from forecaster import PATTERN_NAMES, DAY_LABELS, N_SLOTS  # noqa: E402
from policy_dp import train_policy  # noqa: E402
from recommend import Advisor  # noqa: E402

POLICIES_PKL = os.path.join(_ROOT, "data", "policies.pkl")
BUY_PRICE_RANGE = range(90, 111)          # 官方週日買價 90~110 (閉區間)
PREV_PATTERNS = (None, 0, 1, 2, 3)        # None = 上週波型不知道
QTY_STEP = 10                             # 顆數必為 10 的倍數

# 網頁只呈現 3 種策略 (保守→積極): CVaR → Kelly → Max-Profit; 不顯示 winrate。
STRATEGIES = ("cvar", "kelly", "max_profit")
STRATEGY_INFO = {
    "cvar":       {"label": "穩健保守", "desc": "最保守：壓低最慘 5% 的虧損，下注最小、求穩。"},
    "kelly":      {"label": "Kelly 成長", "desc": "中間：長期複利成長最佳的下注比例，攻守均衡。"},
    "max_profit": {"label": "全力衝刺", "desc": "最積極：期望獲利最大，傾向全押、單週波動也最大。"},
}

# 上週波型按鈕 → 後端 prev_pattern 代碼。(三期=大爆衝/large_spike=1, 四期=小爆衝/small_spike=3)
PREV_PATTERN_CHOICES = (
    ("不知道", None),
    ("波型 (波動)", 0),
    ("四期型 (小爆衝)", 3),
    ("三期型 (大爆衝)", 1),
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
    adv = advisor.advise(buy_price, prev_pattern=prev_pattern,
                         observed=observed, capital=budget)
    fc = adv["forecast"]

    out = {
        "buy_price": buy_price,
        "prev_pattern": prev_pattern,
        "feasible": fc.get("feasible", False),
        "pattern_posterior": {},
        "table": [],
        "remaining_max": None,
        "buy": {},
        "sell": None,
        "current_slot": adv["current_slot"],
    }

    if not fc.get("feasible", False):
        out["error"] = "輸入的價格與官方規則矛盾，請檢查是否輸入錯誤。"
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

    # ---- 週日買入建議: 資金比例 → 顆數 (捨去到 10) ----
    for s in STRATEGIES:
        b = adv["buy"][s]
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
    if adv["sell"] is not None:
        slot = adv["current_slot"]
        cur_p = fc["observed"][slot]
        cur_price = int(cur_p) if cur_p is not None else 0
        sell = {"slot": slot, "price": cur_price}
        for s in STRATEGIES:
            frac = adv["sell"][s]
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
    """單一 slot 的建議賣出顆數 (直接用 policy.decide_sell, 不跑取樣 forecast, 便宜)。
    回傳 (顆數, 文字說明)。holding 為該 slot 當下的持有顆數 (10 倍數)。"""
    if prices[slot] is None or holding <= 0:
        return 0, "—"
    pol = advisor.policy(int(buy_price), prev_pattern)
    filled = _fill_forward(prices[:slot + 1], int(buy_price))
    frac = pol.decide_sell(filled, slot)[strategy]
    if frac >= 0.999:
        return int(holding), "全部賣出"
    if frac <= 1e-6:
        return 0, "不賣 (續抱)"
    return round_down_10(holding * frac), f"賣出 {frac*100:.0f}%"


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
