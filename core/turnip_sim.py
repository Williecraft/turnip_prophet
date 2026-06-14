#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
turnip_sim.py — 動物森友會 大頭菜價格 官方模擬器 (bit-exact)
ACNH Turnip / Stalk Market price simulator.

這是 Nintendo 官方價格演算法的 Python 逐行移植 (port of the datamined
TurnipPrices.cpp by Ninji / Treeki). 給定一個 32-bit 種子 (seed) 與上週的
pattern, 即可 deterministically 產生一週「與官方完全一致」的價格序列。

關鍵正確性要點 (correctness notes):
  1. sead::Random 是 32-bit RNG, 所有運算都用 `& 0xFFFFFFFF` 遮罩。
  2. 遊戲內部以 *單精度 float (float32)* 計算; 為了 bit-exact, 我們用
     f32() 在每一步運算後把值收斂回 float32 (struct round-trip)。
  3. randfloat(a, b) 的「參數順序」會影響同一次抽樣的結果, 必須與官方一致
     (例如 pattern 0 用 randfloat(0.8, 0.6), 不是 (0.6, 0.8))。

來源 (sources): 見 Prerequisite/sources.md
直接執行本檔 (`python turnip_sim.py`) 會跑內建自我驗證。
"""

import struct

# 一週 14 個內部時段槽; index 0,1 = 週日上午/下午 (買菜日, 不賣) -> 永遠 0
# index 2..13 = 週一上午 ~ 週六下午, 共 12 個可賣時段。
SLOTS = 14
SELL_SLOTS = 12

DAY_LABELS = [
    "Mon_AM", "Mon_PM", "Tue_AM", "Tue_PM", "Wed_AM", "Wed_PM",
    "Thu_AM", "Thu_PM", "Fri_AM", "Fri_PM", "Sat_AM", "Sat_PM",
]

PATTERN_NAMES = {
    0: "fluctuating",   # 波動 (高/降/高/降/高)
    1: "large_spike",   # 大爆衝 (先降 -> 2.0~6.0x 尖峰 -> 隨機低)
    2: "decreasing",    # 持續遞減
    3: "small_spike",   # 小爆衝 (降 -> 小尖峰 -> 降)
}


def f32(x: float) -> float:
    """把一個 Python float (float64) 收斂成 float32 的值, 以模擬遊戲的單精度運算。"""
    return struct.unpack("<f", struct.pack("<f", x))[0]


# 預先算好的 float32 常數, 避免每次重算
_F_ONE = f32(1.0)
_F_CEIL = f32(0.99999)


def intceil(x: float) -> int:
    """官方 intceil: (int)(val + 0.99999f) — 以 float32 相加後向 0 截斷。"""
    return int(f32(f32(x) + _F_CEIL))


class SeadRandom:
    """Nintendo sead::Random — 128-bit state 的 xorshift 變體 (32-bit 輸出)。"""

    __slots__ = ("ctx",)

    def __init__(self, seed: int):
        self.init(seed)

    def init(self, seed: int) -> None:
        s = seed & 0xFFFFFFFF
        c = [0, 0, 0, 0]
        c[0] = (0x6C078965 * (s ^ (s >> 30)) + 1) & 0xFFFFFFFF
        c[1] = (0x6C078965 * (c[0] ^ (c[0] >> 30)) + 2) & 0xFFFFFFFF
        c[2] = (0x6C078965 * (c[1] ^ (c[1] >> 30)) + 3) & 0xFFFFFFFF
        c[3] = (0x6C078965 * (c[2] ^ (c[2] >> 30)) + 4) & 0xFFFFFFFF
        self.ctx = c

    def get_u32(self) -> int:
        c = self.ctx
        n = (c[0] ^ ((c[0] << 11) & 0xFFFFFFFF)) & 0xFFFFFFFF
        c[0] = c[1]
        c[1] = c[2]
        c[2] = c[3]
        c[3] = (n ^ (n >> 8) ^ c[3] ^ (c[3] >> 19)) & 0xFFFFFFFF
        return c[3]

    def randint(self, mn: int, mx: int) -> int:
        """官方 randint(min, max) — 閉區間 [mn, mx]。"""
        return ((self.get_u32() * (mx - mn + 1)) >> 32) + mn

    def randfloat(self, a: float, b: float) -> float:
        """官方 randfloat(a, b): a + (mantissa[1,2) - 1) * (b - a), 全程 float32。
        注意 a 可大於 b (例: randfloat(0.8, 0.6)), 分布仍落在 [min, max]。"""
        a = f32(a)
        b = f32(b)
        bits = 0x3F800000 | (self.get_u32() >> 9)
        fval = struct.unpack("<f", struct.pack("<I", bits))[0]  # 落在 [1.0, 2.0)
        return f32(a + f32(f32(fval - _F_ONE) * f32(b - a)))

    def randbool(self) -> bool:
        """官方 randbool: 取最高位元。"""
        return (self.get_u32() & 0x80000000) != 0


def next_pattern(prev_pattern: int, chance: int) -> int:
    """官方 pattern 轉移表: 由上週 pattern + randint(0,99) 決定本週 pattern。
    prev_pattern 為非法值 (>=4, 例如首週未初始化) 時官方回傳 2 (decreasing)。"""
    if prev_pattern < 0 or prev_pattern >= 4:
        return 2
    # thresholds[prev] = [(門檻, pattern), ...], chance < 門檻 即採該 pattern
    thresholds = {
        0: ((20, 0), (50, 1), (65, 2), (100, 3)),
        1: ((50, 0), (55, 1), (75, 2), (100, 3)),
        2: ((25, 0), (70, 1), (75, 2), (100, 3)),
        3: ((45, 0), (70, 1), (85, 2), (100, 3)),
    }[prev_pattern]
    for threshold, pat in thresholds:
        if chance < threshold:
            return pat
    return 3  # 理論上不會到這 (chance < 100 必命中)


def generate_week(prev_pattern: int, seed: int):
    """產生一週價格。回傳 dict:
        base_price : 週日 Daisy Mae 買價 (90~110)
        pattern    : 本週 pattern (0~3)
        prices     : 長度 12 的 list, Mon_AM ~ Sat_PM 的賣價

    RNG 抽樣順序嚴格對齊官方: base_price -> chance -> (各 pattern 的抽樣)。
    """
    rng = SeadRandom(seed)
    base = rng.randint(90, 110)
    chance = rng.randint(0, 99)
    pattern = next_pattern(prev_pattern, chance)

    p = [0] * SLOTS  # index 0,1 = 買菜日, 保持 0

    def price(rate_or_factor):
        return intceil(f32(rate_or_factor * base))

    if pattern == 0:
        # ---- Fluctuating: 高 / 降 / 高 / 降 / 高 ----
        work = 2
        dec_len1 = 3 if rng.randbool() else 2
        dec_len2 = 5 - dec_len1
        hi_len1 = rng.randint(0, 6)
        hi_len2and3 = 7 - hi_len1
        hi_len3 = rng.randint(0, hi_len2and3 - 1)

        for _ in range(hi_len1):
            p[work] = price(rng.randfloat(0.9, 1.4)); work += 1

        rate = rng.randfloat(0.8, 0.6)
        for _ in range(dec_len1):
            p[work] = intceil(f32(rate * base)); work += 1
            rate = f32(rate - 0.04)
            rate = f32(rate - rng.randfloat(0.0, 0.06))

        for _ in range(hi_len2and3 - hi_len3):
            p[work] = price(rng.randfloat(0.9, 1.4)); work += 1

        rate = rng.randfloat(0.8, 0.6)
        for _ in range(dec_len2):
            p[work] = intceil(f32(rate * base)); work += 1
            rate = f32(rate - 0.04)
            rate = f32(rate - rng.randfloat(0.0, 0.06))

        for _ in range(hi_len3):
            p[work] = price(rng.randfloat(0.9, 1.4)); work += 1

    elif pattern == 1:
        # ---- Large spike: 先遞減, 中段爆衝, 之後隨機低 ----
        peak_start = rng.randint(3, 9)
        rate = rng.randfloat(0.9, 0.85)
        work = 2
        while work < peak_start:
            p[work] = intceil(f32(rate * base)); work += 1
            rate = f32(rate - 0.03)
            rate = f32(rate - rng.randfloat(0.0, 0.02))
        p[work] = price(rng.randfloat(0.9, 1.4)); work += 1
        p[work] = price(rng.randfloat(1.4, 2.0)); work += 1
        p[work] = price(rng.randfloat(2.0, 6.0)); work += 1   # 尖峰
        p[work] = price(rng.randfloat(1.4, 2.0)); work += 1
        p[work] = price(rng.randfloat(0.9, 1.4)); work += 1
        while work < SLOTS:
            p[work] = price(rng.randfloat(0.4, 0.9)); work += 1

    elif pattern == 2:
        # ---- Decreasing: 整週持續遞減 ----
        rate = f32(0.9)
        rate = f32(rate - rng.randfloat(0.0, 0.05))
        for work in range(2, SLOTS):
            p[work] = intceil(f32(rate * base))
            rate = f32(rate - 0.03)
            rate = f32(rate - rng.randfloat(0.0, 0.02))

    elif pattern == 3:
        # ---- Small spike: 遞減 -> 小尖峰 -> 遞減 ----
        peak_start = rng.randint(2, 9)
        rate = rng.randfloat(0.9, 0.4)
        work = 2
        while work < peak_start:
            p[work] = intceil(f32(rate * base)); work += 1
            rate = f32(rate - 0.03)
            rate = f32(rate - rng.randfloat(0.0, 0.02))
        p[work] = price(rng.randfloat(0.9, 1.4)); work += 1
        p[work] = price(rng.randfloat(0.9, 1.4)); work += 1
        rate = rng.randfloat(1.4, 2.0)
        p[work] = intceil(f32(rng.randfloat(1.4, rate) * base)) - 1; work += 1
        p[work] = intceil(f32(rate * base)); work += 1            # 尖峰
        p[work] = intceil(f32(rng.randfloat(1.4, rate) * base)) - 1; work += 1
        if work < SLOTS:
            rate = rng.randfloat(0.9, 0.4)
            while work < SLOTS:
                p[work] = intceil(f32(rate * base)); work += 1
                rate = f32(rate - 0.03)
                rate = f32(rate - rng.randfloat(0.0, 0.02))

    return {"base_price": base, "pattern": pattern, "prices": p[2:]}


# --------------------------------------------------------------------------
# 自我驗證 (self-verification): 確認移植正確
# --------------------------------------------------------------------------
def _self_verify(n: int = 200_000) -> None:
    import random as _rnd

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    base_sum = 0
    _rnd.seed(20200320)  # ACNH 發售日, 固定讓驗證可重現

    for _ in range(n):
        prev = _rnd.randint(0, 3)
        seed = _rnd.getrandbits(32)
        w = generate_week(prev, seed)
        base = w["base_price"]
        pat = w["pattern"]
        prices = w["prices"]

        assert 90 <= base <= 110, f"base_price 越界: {base}"
        assert pat in (0, 1, 2, 3)
        assert len(prices) == SELL_SLOTS
        assert all(v > 0 for v in prices), f"出現非正價格: {prices}"
        counts[pat] += 1
        base_sum += base

        if pat == 2:  # decreasing: 必須單調不增
            for i in range(1, len(prices)):
                assert prices[i] <= prices[i - 1], f"decreasing 不單調: {prices}"
        if pat == 1:  # large spike: 尖峰至少 2.0x base
            assert max(prices) >= 2 * base, f"large_spike 尖峰不足: {max(prices)} vs {base}"
        if pat == 3:  # small spike: 必有高於 base 的尖峰
            assert max(prices) > base, f"small_spike 無尖峰: {prices}"

    total = sum(counts.values())
    print(f"[OK] 自我驗證通過, 共 {total} 週")
    print(f"     base_price 平均 = {base_sum / total:.3f} (理論 100.0)")
    print("     pattern 分布:")
    for k in sorted(counts):
        print(f"       {k} {PATTERN_NAMES[k]:<12} {counts[k]:>8}  ({counts[k] / total:6.2%})")


def _demo() -> None:
    print("\n範例 (prev_pattern=2, seed=0x12345678):")
    w = generate_week(2, 0x12345678)
    print(f"  pattern   = {w['pattern']} ({PATTERN_NAMES[w['pattern']]})")
    print(f"  buy price = {w['base_price']}")
    for label, v in zip(DAY_LABELS, w["prices"]):
        print(f"    {label}: {v}")


if __name__ == "__main__":
    _self_verify()
    _demo()
