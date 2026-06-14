#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_app.py — 大頭菜買賣顧問 網頁 UI (Streamlit Cloud 進入點)

布局/風格參考 ac-turnip.com, 另加: 上週波型選擇、策略模式 (含激進度說明)、
週日買入顆數換算、週間逐期賣出建議顆數、預測表格顏色標註、localStorage 記憶 + 清除。
所有演算法都在 core/, 本檔只負責 UI 與互動; 後端呼叫一律走 webapp/advisor_api.py。
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_ROOT, "webapp"), os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from advisor_api import (  # noqa: E402
    advise, get_advisor, slot_sell_qty, round_down_10,
    STRATEGIES, STRATEGY_INFO, PREV_PATTERN_CHOICES,
)

# localStorage (記憶輸入); 缺套件 / 測試環境時自動降級為「只在本次工作階段保留」。
# 設環境變數 TURNIP_NO_LS=1 可強制停用 (AppTest 等無瀏覽器環境會卡在元件 round-trip)。
if os.environ.get("TURNIP_NO_LS"):
    _LS = None
else:
    try:
        from streamlit_local_storage import LocalStorage
        _LS = LocalStorage()
    except Exception:
        _LS = None

N_SLOTS = 12
DAY_ZH = ["週一", "週二", "週三", "週四", "週五", "週六"]
AMPM_ZH = ["上午", "下午"]
PREV_MAP = dict(PREV_PATTERN_CHOICES)            # 標籤 -> 代碼
PATTERN_COLOR = {                                # 波型 -> 主題色
    "fluctuating": "#7fb3d5", "large_spike": "#e8743b",
    "decreasing": "#9aa0a6", "small_spike": "#f2c14e",
}

DEFAULTS = {
    "prev_label": "不知道",
    "strategy": "kelly",
    "buy_price": 0,
    "bought_qty": 0,
    **{f"p_{i}": 0 for i in range(N_SLOTS)},
    **{f"s_{i}": 0 for i in range(N_SLOTS)},
}

st.set_page_config(page_title="大頭菜買賣顧問", page_icon="🥬", layout="centered")


# --------------------------------------------------------------------------
# 狀態持久化 (localStorage): turnip_state = 週資料 (會被清除); turnip_budget = 預算 (不清除)
# --------------------------------------------------------------------------
def _ls_get(key):
    if _LS is None:
        return None
    try:
        return _LS.getItem(key)
    except Exception:
        return None


def _ls_set(key, value):
    if _LS is None:
        return
    try:
        _LS.setItem(key, value, key=f"set_{key}")
    except Exception:
        pass


def init_state():
    ss = st.session_state
    if ss.get("_init"):
        return
    raw = _ls_get("turnip_state")
    try:
        saved = json.loads(raw) if raw else {}
    except Exception:
        saved = {}
    for k, dv in DEFAULTS.items():
        ss[k] = saved.get(k, dv)
    braw = _ls_get("turnip_budget")
    try:
        ss["budget"] = int(braw) if braw else 1_000_000
    except Exception:
        ss["budget"] = 1_000_000
    ss["_init"] = True


def persist_state():
    ss = st.session_state
    _ls_set("turnip_state", json.dumps({k: ss[k] for k in DEFAULTS}))
    _ls_set("turnip_budget", str(int(ss["budget"])))


def clear_data():
    """清除週資料, 保留預算。"""
    ss = st.session_state
    for k, dv in DEFAULTS.items():
        ss[k] = dv


@st.cache_resource(show_spinner="載入策略中…")
def load_advisor():
    return get_advisor()


# --------------------------------------------------------------------------
# 樣式: 海島晨光 / Nook 風 (圓潤、奶油色、葉綠)
# --------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --cream:#fbf4df; --cream2:#f3e8c7; --leaf:#5fa85f; --leaf-d:#3f7a3f;
  --brown:#5b4636; --brown-l:#8a7059; --sky:#8fc7e8; --gold:#f4c95d;
  --turnip:#fff;  --pink:#f6c6d0; --shadow:rgba(95,168,95,.18);
}
html, body, [class*="css"], .stApp{
  font-family:'Noto Sans TC','Fredoka',sans-serif; color:var(--brown);
}
.stApp{
  background:
    radial-gradient(1200px 480px at 50% -8%, #fffaf0 0%, transparent 60%),
    radial-gradient(900px 500px at 110% 10%, #e8f3da 0%, transparent 55%),
    radial-gradient(700px 600px at -10% 30%, #e6f1f7 0%, transparent 50%),
    var(--cream);
}
.block-container{padding-top:2.2rem; max-width:860px;}
h1,h2,h3{font-family:'Fredoka','Noto Sans TC',sans-serif !important; color:var(--brown) !important;}

/* 標題 */
.hero{ text-align:center; margin-bottom:.4rem; }
.hero .title{ font-family:'Fredoka',sans-serif; font-weight:700; font-size:2.5rem;
  color:var(--leaf-d); letter-spacing:.5px; line-height:1.1; }
.hero .sub{ color:var(--brown-l); font-size:.95rem; margin-top:.1rem; }

/* 卡片 */
.card{ background:rgba(255,255,255,.78); border:1.5px solid #ece0bf;
  border-radius:20px; padding:1.1rem 1.3rem; margin:.7rem 0;
  box-shadow:0 8px 24px var(--shadow); backdrop-filter:blur(2px); }
.card h3{ margin:.1rem 0 .7rem; font-size:1.15rem; display:flex; align-items:center; gap:.45rem;}
.sectiontag{ font-size:.78rem; color:#fff; background:var(--leaf);
  padding:.12rem .6rem; border-radius:999px; font-weight:600; }

/* 建議數字大字 */
.big-num{ font-family:'Fredoka',sans-serif; font-weight:700; font-size:1.9rem;
  color:var(--leaf-d); line-height:1.1; }
.muted{ color:var(--brown-l); font-size:.85rem; }
.pill{ display:inline-block; background:var(--cream2); border-radius:999px;
  padding:.12rem .6rem; font-size:.8rem; color:var(--brown); margin-right:.3rem;}

/* radio 變成分段按鈕 */
div[role="radiogroup"]{ gap:.4rem; flex-wrap:wrap; }
div[role="radiogroup"] label{
  background:#fff; border:1.5px solid #e3d7b6; border-radius:14px;
  padding:.35rem .8rem !important; margin:0 !important; transition:.15s; cursor:pointer;}
div[role="radiogroup"] label:hover{ border-color:var(--leaf); transform:translateY(-1px);}
div[role="radiogroup"] label[data-baseweb] div:first-child{ display:none; } /* 藏掉圓點 */

/* 數字輸入 */
.stNumberInput input{ border-radius:12px !important; font-family:'Fredoka',sans-serif;}
.stButton>button{ border-radius:14px; border:1.5px solid var(--leaf);
  background:var(--leaf); color:#fff; font-weight:600; padding:.4rem 1rem;}
.stButton>button:hover{ background:var(--leaf-d); border-color:var(--leaf-d); color:#fff;}

/* 預測表 */
.ftable{ width:100%; border-collapse:separate; border-spacing:0 4px; font-size:.9rem;}
.ftable th{ color:var(--brown-l); font-weight:600; font-size:.8rem; padding:.2rem .4rem; text-align:center;}
.ftable td{ text-align:center; padding:.35rem .4rem; background:#fff;}
.ftable td:first-child{ border-radius:10px 0 0 10px; text-align:left; padding-left:.7rem; font-weight:600;}
.ftable td:last-child{ border-radius:0 10px 10px 0;}
.ftable .now{ outline:2px solid var(--leaf); }
.barwrap{ background:#efe6cb; border-radius:999px; height:12px; overflow:hidden; flex:1;}
.bar{ height:100%; border-radius:999px;}
.prow{ display:flex; align-items:center; gap:.6rem; margin:.25rem 0; font-size:.9rem;}
.prow .pn{ width:6.5rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def fmt(n):
    return f"{int(n):,}"


def cell_color(price, buy):
    """依 價格/買價 比例給格子底色 (參考 ac-turnip: 低=紅, 高=綠, 尖峰=金)。"""
    if price is None or buy in (0, None):
        return "#f3eede", "#9a8f78"
    r = price / buy
    if r < 1.0:
        return "#f3c7c2", "#9c463c"      # 賠錢
    if r < 1.2:
        return "#f7ecc4", "#8a7330"      # 持平
    if r < 1.5:
        return "#dcecb6", "#557a2e"      # 小賺
    if r < 2.2:
        return "#a9dd86", "#2f6b1f"      # 大賺
    return "#f6d873", "#8a5a14"          # 尖峰 (金)


# --------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------
def main():
    inject_css()
    init_state()
    ss = st.session_state
    advisor = load_advisor()

    st.markdown(
        "<div class='hero'><div class='title'>🥬 大頭菜買賣顧問</div>"
        "<div class='sub'>輸入每天的菜價，依官方演算法預測波型並給出買賣建議</div></div>",
        unsafe_allow_html=True,
    )

    # ===== 設定區: 上週波型 / 策略 / 預算 =====
    with st.container():
        st.markdown("<div class='card'><h3><span class='sectiontag'>設定</span> 上週波型 · 策略 · 預算</h3>",
                    unsafe_allow_html=True)

        st.markdown("**上週是什麼波型？**（影響本週波型機率；不確定就選「不知道」）")
        st.radio("上週波型", [c[0] for c in PREV_PATTERN_CHOICES], key="prev_label",
                 horizontal=True, label_visibility="collapsed")

        st.markdown("**選擇策略風格**（保守 → 積極）")
        st.radio("策略", list(STRATEGIES), key="strategy", horizontal=True,
                 format_func=lambda s: STRATEGY_INFO[s]["label"], label_visibility="collapsed")
        st.caption("　".join(
            f"**{STRATEGY_INFO[s]['label']}**：{STRATEGY_INFO[s]['desc']}" for s in STRATEGIES))

        st.number_input("💰 預算（鈴錢，可用來買菜的總額；不會被「清除」清掉）",
                        min_value=0, step=10_000, key="budget")
        st.markdown("</div>", unsafe_allow_html=True)

    prev_pattern = PREV_MAP[ss["prev_label"]]
    strategy = ss["strategy"]
    budget = int(ss["budget"])
    buy_price = int(ss["buy_price"])

    # ===== 星期日: 買入 =====
    with st.container():
        st.markdown("<div class='card'><h3>☀️ 星期日 · 向 Daisy Mae 買菜</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.number_input("本週買入價格（鈴錢/顆，官方 90~110）",
                            min_value=0, max_value=110, step=1, key="buy_price")
        buy_price = int(ss["buy_price"])

        if buy_price and buy_price < 90:
            st.warning("買價通常是 90~110，請確認輸入。")

        if buy_price >= 90:
            res0 = advise(advisor, buy_price, prev_pattern, observed=[], budget=budget, holding=0)
            b = res0["buy"][strategy]
            with c2:
                st.markdown(
                    f"<div class='muted'>建議買入（{STRATEGY_INFO[strategy]['label']}）</div>"
                    f"<div class='big-num'>{fmt(b['qty'])} 顆</div>"
                    f"<div class='muted'>約 {fmt(b['bells'])} 鈴 · {b['tag']}</div>",
                    unsafe_allow_html=True)
            others = "　".join(
                f"<span class='pill'>{STRATEGY_INFO[s]['label']} {fmt(res0['buy'][s]['qty'])} 顆</span>"
                for s in STRATEGIES)
            st.markdown(f"<div style='margin-top:.4rem'>{others}</div>", unsafe_allow_html=True)
            st.number_input("✅ 我實際買了幾顆（10 的倍數）",
                            min_value=0, step=10, key="bought_qty")
        else:
            st.info("輸入週日買價後，這裡會顯示各策略的建議買入顆數與鈴錢。")
        st.markdown("</div>", unsafe_allow_html=True)

    bought_qty = round_down_10(ss["bought_qty"])

    # 讀取週間價格
    prices = [(int(ss[f"p_{i}"]) if int(ss[f"p_{i}"]) > 0 else None) for i in range(N_SLOTS)]
    known = [i for i, v in enumerate(prices) if v is not None]
    cur_slot = max(known) if known else None

    # ===== 週間: 賣出 (12 時段) =====
    with st.container():
        st.markdown("<div class='card'><h3>🛒 週一～週六 · 在商店賣菜</h3>", unsafe_allow_html=True)
        if not bought_qty:
            st.info("先在上方填「實際買了幾顆」，才能算出每期建議賣出顆數。")

        st.markdown(
            "<div class='prow muted'><div class='pn'>時段</div>"
            "<div style='width:7rem'>菜價</div><div style='width:8rem'>建議賣出</div>"
            "<div style='width:8rem'>我實際賣出</div></div>", unsafe_allow_html=True)

        holding = bought_qty
        for i in range(N_SLOTS):
            day, ap = DAY_ZH[i // 2], AMPM_ZH[i % 2]
            sugg_qty, sugg_txt = (0, "—")
            if buy_price >= 90 and prices[i] is not None and holding > 0:
                sugg_qty, sugg_txt = slot_sell_qty(
                    advisor, buy_price, prev_pattern, prices, i, holding, strategy)

            col = st.columns([1.1, 1.2, 1.4, 1.4])
            col[0].markdown(f"**{day} {ap}**"
                            + ("　🟢" if i == cur_slot else ""))
            col[1].number_input(f"price_{i}", min_value=0, max_value=999, step=1,
                                key=f"p_{i}", label_visibility="collapsed")
            tag = f"**{fmt(sugg_qty)}** 顆" if prices[i] is not None and bought_qty else "—"
            col[2].markdown(f"{tag}<br><span class='muted'>{sugg_txt}</span>", unsafe_allow_html=True)
            col[3].number_input(f"sold_{i}", min_value=0, step=10,
                                key=f"s_{i}", label_visibility="collapsed")

            actual_sold = min(round_down_10(ss[f"s_{i}"]), holding)
            holding -= actual_sold

        st.markdown(
            f"<div class='muted' style='margin-top:.5rem'>目前剩餘持有："
            f"<b>{fmt(max(holding,0))}</b> 顆</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== 預測區: 波型機率 + 價格表 =====
    if buy_price >= 90:
        observed = prices[:cur_slot + 1] if cur_slot is not None else []
        res = advise(advisor, buy_price, prev_pattern, observed=observed, budget=budget, holding=0)
        with st.container():
            st.markdown("<div class='card'><h3>🔮 預測（依目前已知資訊）</h3>", unsafe_allow_html=True)
            if not res["feasible"]:
                st.error(res.get("error", "輸入的價格與官方規則矛盾，請檢查。"))
            else:
                # 波型後驗機率長條
                st.markdown("**本週波型機率**", unsafe_allow_html=True)
                post = res["pattern_posterior"]
                zh = {"fluctuating": "波動", "large_spike": "大爆衝(三期)",
                      "decreasing": "遞減", "small_spike": "小爆衝(四期)"}
                for name in ["large_spike", "small_spike", "fluctuating", "decreasing"]:
                    p = post.get(name, 0.0)
                    st.markdown(
                        f"<div class='prow'><div class='pn'>{zh[name]}</div>"
                        f"<div class='barwrap'><div class='bar' style='width:{p*100:.0f}%;"
                        f"background:{PATTERN_COLOR[name]}'></div></div>"
                        f"<div style='width:3rem;text-align:right'>{p*100:.0f}%</div></div>",
                        unsafe_allow_html=True)

                rm = res["remaining_max"]
                st.markdown(
                    f"<div class='muted' style='margin:.5rem 0'>剩餘最高價預估："
                    f"中位 <b>{fmt(rm['q50'])}</b> · 期望 {fmt(rm['mean'])} 鈴/顆</div>",
                    unsafe_allow_html=True)

                # 價格表
                rows = ["<table class='ftable'><tr><th>時段</th><th>你的價格</th>"
                        "<th>預估中位</th><th>可能區間</th><th>保證範圍</th></tr>"]
                for r in res["table"]:
                    day, ap = DAY_ZH[r["slot"] // 2], AMPM_ZH[r["slot"] % 2]
                    is_obs = r["obs"] is not None
                    show_price = r["obs"] if is_obs else r["q50"]
                    bg, fg = cell_color(show_price, buy_price)
                    nowcls = " now" if r["slot"] == cur_slot else ""
                    obs_txt = f"<b>{r['obs']}</b>" if is_obs else "—"
                    rows.append(
                        f"<tr><td>{day}{ap}</td>"
                        f"<td class='{nowcls.strip()}'>{obs_txt}</td>"
                        f"<td style='background:{bg};color:{fg};font-weight:700;border-radius:8px'>{r['q50']}</td>"
                        f"<td class='muted'>{r['q10']}–{r['q90']}</td>"
                        f"<td class='muted'>{r['gmin']}–{r['smax']}</td></tr>")
                rows.append("</table>")
                st.markdown("".join(rows), unsafe_allow_html=True)
                st.caption("顏色：🟥 賠錢　🟨 持平　🟩 賺　🟧 尖峰。已輸入的時段以你的價格為準。")
            st.markdown("</div>", unsafe_allow_html=True)

    # ===== 清除 =====
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    cL, cR = st.columns([3, 1])
    cR.button("🧹 清除資料", on_click=clear_data, use_container_width=True,
              help="清除本週所有買賣輸入（預算會保留）")
    cL.caption("資料會自動記在這台裝置的瀏覽器，下次打開免重填。")

    persist_state()


if __name__ == "__main__":
    main()
