# 動森大頭菜 / ACNH Turnip (Stalk Market)

目標: 在《動物森友會》大頭菜上賺大錢。本 repo 第一階段 = **官方規則蒐集 + bit-exact 模擬器**,
作為日後「找出最佳買賣規則」的資料引擎。

## 內容 (Contents)

```
Prerequisite/
  ALGORITHM.md              官方演算法詳解 (RNG / 機率表 / 四 pattern 公式)
  TurnipPrices_official.cpp 官方演算法 C++ 參考實作 (對照用)
  sources.md                所有來源連結
# 第一階段: 官方模擬器 + 資料
turnip_sim.py               官方模擬器 (核心, 純標準庫, bit-exact)
generate_dataset.py         批次生成資料 -> parquet / csv
# 第二階段: 預測 + 決策 + 評估 (買賣顧問)
forecaster.py               機率預測引擎 (波型後驗 + 價格區間 + 剩餘最高價分布)
policy_dp.py                DP 決策 (4 套策略: max_profit/kelly/winrate/cvar)
backtest.py                 大規模評估 (對 bit-exact 真實週 out-of-sample)
recommend.py                買賣顧問 (app 核心介面; 未來網頁 UI 後端)
requirements.txt            numpy / pyarrow / pandas
```

## 買賣顧問怎麼用 (recommend.py)

```python
from recommend import Advisor
adv = Advisor(base_qty=10000)
a = adv.advise(buy_price=96, prev_pattern=2, observed=[84, 105], capital=2_000_000)
adv.print_advice(a)   # 印出: 波型機率 + 價格區間 + 四面向的買/賣建議
```

四個面向 (並列給你選):
- **max_profit** 最高獲利 (風險中性, EV 最佳停止, all-in)
- **kelly** 成長最佳 (賣出同 max_profit; 買入用 full-Kelly 部位, 長期複利最佳)
- **winrate** ①勝率/賠最少 (出現保本價即落袋)
- **cvar** ②尾部折衷 (half-Kelly 買 + 風險調整賣, 砍尾部風險)

決策變數: 買=資金比例 f (1=all-in); 賣=賣出佔目前持有比例。顆數/鈴錢為 UI 換算。

### 實測表現 (out-of-sample, bit-exact 真實週, 上週未知)
| 策略 | 平均報酬 | 勝率 | 最壞 | CVaR5 |
|---|---|---|---|---|
| oracle (完美後見) | +103% | 85% | −14% | −14% |
| **max_profit** | **+54%** | 73% | −84% | −61% |
| **cvar** | +42% | 77% | −66% | **−48%** |
| **winrate** | +11% | **85%** | −65% | −59% |
| greedy 門檻 | +29% | 82% | −65% | −59% |
| naive (週六硬賣) | −25% | 24% | −86% | −69% |

## 這是什麼 / 為什麼可信

`turnip_sim.py` 是 Nintendo 官方價格演算法 (社群 datamine, 由 Ninji/Treeki 反組譯) 的
Python 逐行移植。給一個 32-bit 種子 + 上週 pattern, 就能 deterministically 產生一週
「與官方完全一致」的價格。為達 bit-exact:

- `sead::Random` 全程 32-bit 遮罩;
- 以 `f32()` 模擬遊戲的單精度 float 運算;
- `randfloat` 參數順序、`intceil` 取整完全照官方。

> **不需要島嶼種子**: 生成資料只要自行枚舉/隨機抽種子即可, 每個種子 = 一週合法官方價格。

## 快速開始 (Quick start)

```bash
# 1) 驗證模擬器 (純標準庫, 免安裝)；核心模組都在 core/
python core/turnip_sim.py

# 2) 安裝套件 (numpy 必需; pyarrow/pandas 供資料輸出; streamlit 供網頁 UI)
pip install -r requirements.txt

# 3) 生成資料 (100 萬週, 每週獨立+上週波型當條件 -> parquet)
python core/generate_dataset.py --n 1000000 --out data/weeks.parquet
```

## 大頭菜機制 (重點)

- **買**: 只能週日上午向 Daisy Mae 買, 價格隨機 90~110 鈴/顆, 一週一次。
- **賣**: 週一~週六、每天上午/下午共 **12 時段** 在 Nook 商店賣; **過週六全爛歸零**。
- **四種 pattern**: 0 波動 / 1 大爆衝(2~6×) / 2 遞減 / 3 小爆衝(1.4~2×)。詳見 ALGORITHM.md。

## 跨週相依: 只有「波型」會傳遞, 價格不會

官方規則裡**上週波型會影響本週波型**(4×4 轉移機率表), 但**上週價格對本週價格毫無影響**
(memoryless), `base_price` 每週重抽。所以每週**獨立生成 + 把上週波型當條件欄位 `prev_pattern`**
就足夠且正確 —— 這是 `generate_dataset.py` 的**預設 (independent) 模式**, `prev_pattern`
預設均勻抽 0~3, 讓四種「上週波型」條件都有充足樣本 (最利於學條件式買賣規則)。

另有 `--mode chain` (連鎖) 僅在想估「長期整體遇到各波型的頻率」(stationary 邊際分布) 時才用。

## 資料欄位 (generate_dataset.py 輸出)

`week_index, seed, prev_pattern, base_price, pattern, price_Mon_AM..price_Sat_PM (12),
max_price, max_price_slot, peak_multiplier`

每列一週; `prev_pattern` = 上週波型 (生成條件); `peak_multiplier = max_price / base_price`
即該週報酬率上限。

## 驗證 (Verification)

```bash
python core/turnip_sim.py     # 模擬器自我驗證 (bit-exact 不變式)
python core/forecaster.py     # 預測引擎範例
python core/policy_dp.py      # 策略自我測試 (EV 達 oracle ~86%, 勝過 naive ~3x)
python core/backtest.py       # 大規模 out-of-sample 評估 + forecaster 校準 (~100s)
python core/recommend.py      # 端到端買賣顧問示範
```

## 網頁 UI (Streamlit Cloud)

把 `recommend.py` 的 `Advisor` 包成網頁 (類 ac-turnip.com, 多了上週波型輸入、策略模式選擇與
「現在價格賣幾顆」)。為避免現場訓練，先離線把每個 (買價, 上週波型) 的策略序列化成
`data/policies.pkl`，UI 直接查表。

```bash
python webapp/precompute.py        # 離線預訓練所有策略 -> data/policies.pkl
streamlit run streamlit_app.py     # 本機預覽
```

部署：推到 GitHub，於 [share.streamlit.io](https://share.streamlit.io) 指定 `streamlit_app.py`
為進入點即可（免費）。`data/policies.pkl` 一併進 repo，雲端啟動就不必訓練。

- `webapp/advisor_api.py` — 薄包裝層：比例→顆數/鈴錢換算、強制 10 的倍數、部位追蹤。
- `webapp/precompute.py` — 離線預訓練所有 (買價 90~110 × 上週波型 5 種) 策略。
