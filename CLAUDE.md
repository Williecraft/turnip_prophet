# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

專案語言為繁體中文（程式碼註解、docstring、輸出皆為中文），請以繁體中文回應與撰寫註解。

## 這是什麼

《動物森友會》大頭菜（Stalk Market）買賣顧問。從官方價格演算法的 bit-exact 模擬器出發，
建出「機率預測 → 決策 → 大規模評估 → 顧問介面」的管線。無外部服務、無前端，純 Python。

## 常用指令

無 build / lint 步驟。核心模組都在 `core/`，直接執行即跑自我測試 / 範例
（直接跑 `python core/xxx.py` 時，Python 會把 `core/` 放進 sys.path，故各模組維持扁平
`from turnip_sim import …` 的寫法即可運作，搬進 core/ 後**不需改 import**）：

```bash
python core/turnip_sim.py   # 模擬器 bit-exact 自我驗證 (20 萬週不變式)
python core/forecaster.py   # 預測引擎範例輸出
python core/policy_dp.py    # 策略自我測試 (EV 達 oracle ~86%)
python core/backtest.py     # out-of-sample 大規模評估 + 校準 (~100s)
python core/recommend.py    # 端到端顧問示範

pip install -r requirements.txt   # numpy 必需; pyarrow/pandas 僅 generate_dataset 輸出 parquet 需要; streamlit 為網頁 UI
python core/generate_dataset.py --n 1000000 --out data/weeks.parquet

# 網頁 UI (Streamlit Cloud)
python webapp/precompute.py        # 離線預訓練所有策略 -> data/policies.pkl
streamlit run streamlit_app.py     # 本機預覽顧問網頁
```

沒有測試框架（pytest 等）；「測試」=各模組 `__main__` 的 assert 自我驗證。要驗證改動，
直接跑對應模組即可。環境為 Windows / PowerShell。

## 架構（資料流）

```
turnip_sim.py   官方演算法 bit-exact 移植 (唯一「真相來源」, 純標準庫)
      │ generate_week(prev_pattern, seed) -> 一週 12 個賣價
      ├──────────────► generate_dataset.py   批次落地成 parquet/csv (策略挖掘用)
      │
forecaster.py   信念引擎: 列舉「子波型」(類 Turnip Prophet), 用觀測篩可行解
      │ forecast() -> 波型後驗 + per-slot 價格分布 + 剩餘最高價分布
      ▼
policy_dp.py    決策引擎: 對每個 (買價, 上週波型) 訓練多套策略
      │ train_policy() -> Policy{ decide_sell(), recommend_buy() }
      ▼
recommend.py    Advisor: 串 forecaster + policy_dp, 快取已訓練策略, 人類可讀輸出
                (= 網頁 UI 的後端)
      │
webapp/advisor_api.py   網頁薄包裝: 呼叫 Advisor, 把「資金比例/賣出比例」換算成
      │                 顆數/鈴錢, 強制 10 的倍數取整, 並追蹤使用者的實際買賣部位。
      │                 載入 data/policies.pkl (precompute.py 預訓練) 免現場訓練。
      ▼
streamlit_app.py        Streamlit Cloud 進入點 (= 前端 UI)

backtest.py     誠實評估: 策略用 forecaster 的取樣器訓練, 但測試改用 turnip_sim
                的真實週 (out-of-sample), 確認能類化且勝過基準線。
```

**目錄結構**：`core/` 核心引擎（上面 6 支，bit-exact 區，import 維持扁平）；
`webapp/` 網頁專屬程式（`advisor_api.py` 包裝層、`precompute.py` 離線預訓練）；
`data/` 產物（`policies.pkl` 策略快取、parquet 資料集）；`streamlit_app.py` 在根目錄
（Streamlit Cloud 進入點）；`Prerequisite/` 官方參考實作與演算法文件；`tmp/` 暫存實驗。
非 core 的程式要 import 核心時，於檔案開頭把 `core/` 加進 `sys.path` 再 `from recommend import …`。

**兩套引擎的「誠實原則」**：策略由 `forecaster` 的子波型取樣器（numpy、統計近似）訓練，
但 `backtest.py` 一律用**獨立的** `turnip_sim`（bit-exact）真實週做 out-of-sample 測試。
改了取樣器或策略後，務必跑 `backtest.py` 確認在真實週上仍成立——不要只信訓練分數。

## 必守不變式（改 turnip_sim.py 時最關鍵）

bit-exact 移植非常脆弱，破壞任一點就不再與官方一致：

- 全程 32-bit：`sead::Random` 所有運算 `& 0xFFFFFFFF`。
- 單精度：每步用 `f32()`（struct round-trip）收斂回 float32，模擬遊戲的 float 運算。
- `randfloat(a, b)` 的**參數順序**會影響抽樣結果，且 a 可大於 b（例 `randfloat(0.8, 0.6)`），必須照官方。
- RNG 抽樣**順序**固定：base_price → chance → 各 pattern 內部抽樣。
- `intceil` = `(int)(val + 0.99999f)`，以 float32 相加。
- 非法 `prev_pattern`（>=4，如首週）官方回傳 pattern 2 (decreasing)。

對照實作見 `Prerequisite/TurnipPrices_official.cpp`；演算法詳解見 `Prerequisite/ALGORITHM.md`。

## 重要慣例與易踩雷

- **Slot 索引**：`turnip_sim` 內部用 work 槽 0..13（0,1=週日買菜日恆為 0）；對外輸出與
  `forecaster`/`policy_dp` 統一用 slot 0..11（= work − 2，Mon_AM..Sat_PM）。`N_SLOTS = 12`。
- **跨週只傳「波型」，不傳價格**：官方上週波型經 4×4 轉移表影響本週波型，但價格 memoryless，
  `base_price` 每週重抽。故資料預設 `independent` 模式（每週獨立 + `prev_pattern` 當條件欄位）；
  `chain` 模式僅用於估長期 stationary 邊際分布。`prev_pattern=None` 代表上週未知，用 stationary 先驗。
- **轉移機率有兩份**：`turnip_sim.next_pattern` 用官方門檻表（整數 chance），
  `forecaster.TRANSITION` 是其機率矩陣化。改一邊要同步另一邊。
- **策略清單以程式碼為準**：策略有 **4 個**——
  `policy_dp.OBJECTIVES = ("max_profit", "kelly", "cvar", "winrate")` 是唯一可信來源。
  動策略時請一併更新 README 與各 docstring 的文字描述。
- **策略風險光譜**（保守→積極）：winrate(¼-Kelly) → cvar(½-Kelly) → kelly(full-Kelly) →
  max_profit(all-in)。決策變數：買=資金比例 f（1=all-in）；賣=賣出佔目前持有比例。
  max_profit 與 kelly 賣出邏輯相同，差別只在買入部位大小。
