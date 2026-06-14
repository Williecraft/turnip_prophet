# 來源 (Sources)

ACNH 大頭菜價格官方演算法的比對與蒐集來源。標 ★ 者為權威/一手來源。

## 一手 / 權威 (datamine)

- ★ **Ninji / Treeki — TurnipPrices.cpp (Gist)**
  反組譯自遊戲本體的官方價格演算法, 業界所有計算機/預測器的源頭。
  https://gist.github.com/Treeki/85be14d297c80c8b3c0a76375743325b

- ★ **simontime/Resead** — `sead::Random` 的還原實作 (Nintendo 自家 RNG)。
  https://github.com/simontime/Resead

- ★ **Mike Bryant — ac-nh-turnip-prices** (Turnip Prophet 後端, 維護版)
  完整 pattern 機率與 JS 移植, 採貝氏推論做預測 (而非破解種子)。
  https://github.com/mikebryant/ac-nh-turnip-prices
  鏡像/分支: https://github.com/szymczdm/ac-nh-turnip-prices

## 規則白話說明 / 交叉驗證

- **Nookipedia — Stalk Market**
  買賣機制、四種 pattern 的白話說明。
  https://nookipedia.com/wiki/Stalk_Market

- **Turnip Prophet** (線上預測器, 用於 bit-exact 對照)
  https://turnipprophet.io/

## 取得「自己島嶼種子」的工具 (本階段未用, 留存備查)

> 本專案「生成資料 / 找規則」不需要真實種子; 以下僅供日後若要預測自己島嶼時參考。
> 注意: 從週價格反推種子實務上難以唯一決定 (intceil/randfloat 取整丟失位元),
> 這也是主流預測器改用貝氏推論的原因。

- **NHSE** (kwsch) — 存檔編輯器, 可讀/改 Turnip Exchange (需破解主機)。
  https://github.com/kwsch/NHSE
  NHSE ↔ Meteonook 種子換算: `meteonook = nhse - 2147483648`
- **averne/Turnips** — Switch homebrew, 主機上直接顯示本週大頭菜價格。
  https://github.com/averne/Turnips
