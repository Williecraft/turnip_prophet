#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend.py — 大頭菜買賣顧問 (app 核心介面; 未來網頁 UI 的後端)

把 forecaster (預測) 與 policy_dp (決策) 串成單一介面:
  輸入: 週日買價、上週波型(可選)、目前為止的逐期價格、(可選)資金、基礎量。
  輸出: 波型機率 + 未來價格區間 + 四個面向的『這期賣多少』與『週日買多少』。

四個面向:
  max_profit 最高獲利(風險中性) / kelly 成長最佳(full-Kelly) /
  winrate ①勝率(賠最少) / cvar ②尾部風險折衷

決策變數: 買=資金比例 f(0~1, 1=all-in); 賣=賣出佔『目前持有』比例。
顆數/鈴錢=純 UI 換算 (顆數=倍數x10000, 鈴錢=顆數x買價)。
"""

from __future__ import annotations

from forecaster import forecast, print_forecast, PATTERN_NAMES, DAY_LABELS
from policy_dp import train_policy, OBJECTIVES

OBJ_LABEL = {
    "max_profit": "最高獲利",
    "kelly": "Kelly成長最佳",
    "winrate": "①勝率(賠最少)",
    "cvar": "②尾部折衷(CVaR)",
}


class Advisor:
    """快取各 (買價, 上週波型) 的訓練好策略。"""

    def __init__(self, base_qty=10000):
        self.base_qty = base_qty
        self._cache = {}

    def policy(self, buy_price, prev_pattern):
        key = (int(buy_price), prev_pattern)
        if key not in self._cache:
            self._cache[key] = train_policy(int(buy_price), prev_pattern)
        return self._cache[key]

    def advise(self, buy_price, prev_pattern=None, observed=(), capital=None):
        observed = list(observed)
        fc = forecast(buy_price, observed, prev_pattern=prev_pattern)
        pol = self.policy(buy_price, prev_pattern)
        buy = pol.recommend_buy(capital=capital, base_qty=self.base_qty)
        sell = None
        if observed:
            t = len(observed) - 1
            sell = pol.decide_sell(observed, t)   # objective -> 賣出佔目前持有比例
        return {"forecast": fc, "buy": buy, "sell": sell,
                "buy_price": int(buy_price), "prev_pattern": prev_pattern,
                "current_slot": (len(observed) - 1) if observed else None}

    # ---- 人類可讀輸出 ----
    def print_advice(self, adv):
        print(f"\n{'='*64}")
        print(f"買價={adv['buy_price']}  上週波型="
              f"{adv['prev_pattern']}({PATTERN_NAMES.get(adv['prev_pattern'],'未知') if adv['prev_pattern'] is not None else '未知'})")
        print_forecast(adv["forecast"])

        print("\n● 週日買入建議 (資金比例 f; 顆數/鈴錢為換算):")
        for o in OBJECTIVES:
            b = adv["buy"][o]
            tag = "ALL-IN" if b["all_in"] else f"{b['capital_fraction']*100:.0f}% 資金"
            extra = ""
            if "quantity" in b:
                extra = f"  → {b['quantity']} 顆 / {b['bells']:,} 鈴 (×{b['multiplier_of_base']:.2f} 基礎量)"
            print(f"   {OBJ_LABEL[o]:<16} {tag}{extra}")

        if adv["sell"] is not None:
            slot = DAY_LABELS[adv["current_slot"]]
            price = adv["forecast"]["observed"][adv["current_slot"]]
            print(f"\n● 本期賣出建議 ({slot} 價格={price}; 賣出佔目前持有比例):")
            for o in OBJECTIVES:
                f = adv["sell"][o]
                act = "全部賣出" if f >= 0.999 else ("不賣 (續抱)" if f <= 1e-6 else f"賣出 {f*100:.0f}%")
                print(f"   {OBJ_LABEL[o]:<16} {act}")
        else:
            print("\n(尚未輸入任何賣價; 上方為週日買入決策)")


def _demo():
    """端到端 smoke: 模擬一個真實『大爆衝』週, 逐期輸入價格看建議變化。"""
    from turnip_sim import generate_week
    adv = Advisor(base_qty=10000)

    prev, seed = 2, 0x12345678
    w = generate_week(prev, seed)   # 取一個真實週當輸入腳本
    prices = w["prices"]
    print(f"[示範用真實週] 實際波型={w['pattern']}({PATTERN_NAMES[w['pattern']]}) "
          f"買價={w['base_price']} 實際價格={prices}")

    a = adv.advise(w["base_price"], prev_pattern=prev, observed=[], capital=2_000_000)
    adv.print_advice(a)

    for k in (1, 3, 5):   # 看到第 1、3、5 期後的建議
        a = adv.advise(w["base_price"], prev_pattern=prev, observed=prices[:k], capital=2_000_000)
        adv.print_advice(a)


if __name__ == "__main__":
    _demo()
