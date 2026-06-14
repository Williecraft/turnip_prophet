# ACNH 大頭菜價格 官方演算法 (datamined)

> Animal Crossing: New Horizons — Turnip / Stalk Market price algorithm.
> 本文件整理自遊戲本體反組譯 (datamine) 出的官方演算法, 供 `turnip_sim.py` 逐行對照。
> 來源見 [sources.md](./sources.md)。原始 C++ 見 [TurnipPrices_official.cpp](./TurnipPrices_official.cpp)。

---

## 0. 名詞 / 機制 (mechanics)

- 一週 = 14 個「半天」時段槽 (slots):
  - slot 0, 1 = **週日上午 / 下午** → 向 Daisy Mae **買** 大頭菜, 不可賣, 價格欄位為 0。
  - slot 2 ~ 13 = **週一上午 ~ 週六下午**, 共 **12 個可賣時段** (Nook 商店買回)。
- **買價 (base price)**: 每週週日隨機 `randint(90, 110)` 鈴錢/顆, 一週只買得到一次。
- **賣價**: 由本週 pattern + base price 決定 12 個時段價格。**過週六全部腐爛歸零**。
- 玩家**無法影響**價格; 一切由開週時的 RNG 決定。

---

## 1. 亂數產生器 sead::Random

Nintendo 自家 RNG, 128-bit 狀態的 xorshift 變體, 每次輸出 32-bit。**所有運算為 32-bit 無號**。

```
init(seed):
    ctx[0] = 0x6C078965 * (seed   ^ (seed   >> 30)) + 1
    ctx[1] = 0x6C078965 * (ctx[0] ^ (ctx[0] >> 30)) + 2
    ctx[2] = 0x6C078965 * (ctx[1] ^ (ctx[1] >> 30)) + 3
    ctx[3] = 0x6C078965 * (ctx[2] ^ (ctx[2] >> 30)) + 4

getU32():
    n = ctx[0] ^ (ctx[0] << 11)
    ctx[0..2] = ctx[1..3]                       # 左移
    ctx[3] = n ^ (n >> 8) ^ ctx[3] ^ (ctx[3] >> 19)
    return ctx[3]
```

衍生函式:

| 函式 | 定義 | 說明 |
|------|------|------|
| `randint(mn, mx)` | `((u64)getU32() * (mx-mn+1)) >> 32 + mn` | 閉區間 `[mn, mx]` |
| `randfloat(a, b)` | `a + (mantissa - 1) * (b - a)` | `mantissa` = `0x3F800000 \| (getU32()>>9)` 解讀成 float, 落在 `[1,2)` |
| `randbool()` | `getU32() & 0x80000000` | 取最高位元 |
| `intceil(v)` | `(int)(v + 0.99999f)` | 向上取整 (正數時) |

> ⚠️ **bit-exact 關鍵 1**: 遊戲全程用 **單精度 float32** 計算。Python 預設 float64, 因此
> `turnip_sim.py` 用 `f32()` 在每步運算後收斂回 float32。
> ⚠️ **bit-exact 關鍵 2**: `randfloat(a, b)` 的**參數順序會改變結果**。`a` 可以大於 `b`
> (如 pattern 0 的 `randfloat(0.8, 0.6)`); 分布範圍相同但同一抽樣值不同, 必須照官方順序。

---

## 2. 開週流程

```
base       = randint(90, 110)        # 買價
chance     = randint(0, 99)          # 決定 pattern
pattern    = nextPattern(上週pattern, chance)
依 pattern 產生 12 個賣價
```

### Pattern 轉移機率表

`chance` < 門檻 即採該 pattern (由左到右第一個命中)。上週 pattern 非法 (≥4, 例如全新存檔
未初始化) 時 → 強制 pattern 2。

| 上週 \ chance | →0 (fluct) | →1 (large) | →2 (decr) | →3 (small) |
|---|---|---|---|---|
| **0 fluctuating** | <20 | <50 | <65 | else |
| **1 large spike** | <50 | <55 | <75 | else |
| **2 decreasing**  | <25 | <70 | <75 | else |
| **3 small spike** | <45 | <70 | <85 | else |

> **首週特例**: 官方原始碼有一段以 `FirstKabuBuy` 旗標處理「玩家第一次買菜」的邏輯, 但該段
> 在 datamine 版本中是**註解掉的**, 實際運行採上表。主流預測器 (Turnip Prophet 等) 亦
> 「不支援第一週」。本專案資料生成預設略過首週情境。

---

## 3. 四種 Pattern 公式

下面 `rate` 為 float32, `base` 為買價; `price = intceil(rate * base)`。

### Pattern 0 — Fluctuating (波動: 高/降/高/降/高)
```
decLen1 = randbool() ? 3 : 2;   decLen2 = 5 - decLen1
hiLen1  = randint(0, 6)
hiLen2and3 = 7 - hiLen1
hiLen3  = randint(0, hiLen2and3 - 1)

高峰段1: hiLen1 次   price = intceil(randfloat(0.9, 1.4) * base)
遞減段1: rate = randfloat(0.8, 0.6)
         decLen1 次:  price = intceil(rate*base); rate -= 0.04; rate -= randfloat(0, 0.06)
高峰段2: (hiLen2and3 - hiLen3) 次  intceil(randfloat(0.9, 1.4) * base)
遞減段2: rate = randfloat(0.8, 0.6)
         decLen2 次:  同遞減段1
高峰段3: hiLen3 次   intceil(randfloat(0.9, 1.4) * base)
```

### Pattern 1 — Large Spike (大爆衝)
```
peakStart = randint(3, 9)
rate = randfloat(0.9, 0.85)
work=2 .. peakStart-1: price = intceil(rate*base); rate -= 0.03; rate -= randfloat(0, 0.02)
尖峰 5 連:
    intceil(randfloat(0.9, 1.4) * base)
    intceil(randfloat(1.4, 2.0) * base)
    intceil(randfloat(2.0, 6.0) * base)   ← 最高點
    intceil(randfloat(1.4, 2.0) * base)
    intceil(randfloat(0.9, 1.4) * base)
其餘到 slot 13: intceil(randfloat(0.4, 0.9) * base)
```

### Pattern 2 — Decreasing (持續遞減)
```
rate = 0.9; rate -= randfloat(0, 0.05)
work=2 .. 13: price = intceil(rate*base); rate -= 0.03; rate -= randfloat(0, 0.02)
```

### Pattern 3 — Small Spike (小爆衝)
```
peakStart = randint(2, 9)
rate = randfloat(0.9, 0.4)
work=2 .. peakStart-1: price = intceil(rate*base); rate -= 0.03; rate -= randfloat(0, 0.02)
尖峰:
    intceil(randfloat(0.9, 1.4) * base)
    intceil(randfloat(0.9, 1.4) * base)
    rate = randfloat(1.4, 2.0)
    intceil(randfloat(1.4, rate) * base) - 1
    intceil(rate * base)                   ← 最高點
    intceil(randfloat(1.4, rate) * base) - 1
其餘 (若還有 slot): rate = randfloat(0.9, 0.4)
    price = intceil(rate*base); rate -= 0.03; rate -= randfloat(0, 0.02)
```

---

## 4. 對策略的意涵 (給下一階段)

- **買**: 只有週日一次, 買價 90~110。買價越低, 同樣的 pattern 報酬率越高。
- **賣**: 看時段價格決定出手; 重點是辨識/賭 pattern。
  - large spike 可達 **2.0~6.0×** 買價 → 最大獲利來源。
  - small spike 約 **1.4~2.0×**。
  - decreasing 必賠 → 早賣停損。
  - fluctuating 高峰段約 **0.9~1.4×**。
- 因價格完全由開週 RNG 決定, 「最佳賣出規則」可由本專案模擬器生成的海量資料統計得出。
