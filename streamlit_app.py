#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_app.py — 大頭菜買賣顧問 網頁 UI (Streamlit Cloud 進入點)

版面參考 ac-turnip.com: 上方價格輸入 + 區間圖表 + 「波型 × 時間」價格表;
另加本站特色: 上週波型選擇、策略模式 (4 種, 保守→積極)、週日買入顆數、週間賣出顆數。
所有演算法都在 core/, 本檔只負責 UI; 後端一律走 webapp/advisor_api.py。
"""

from __future__ import annotations

import json
import os
import sys

import plotly.graph_objects as go
import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_ROOT, "webapp"), os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from advisor_api import (  # noqa: E402
    advise, get_advisor, slot_sell_qty, pattern_table, round_down_10,
    STRATEGIES, STRATEGY_INFO, PREV_PATTERN_CHOICES,
)

# localStorage (記憶輸入); 缺套件 / 測試環境時降級為「只在本次工作階段保留」。
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
DAY_SHORT = ["一", "二", "三", "四", "五", "六"]
AMPM = ["上午", "下午"]
AMPM_S = ["上", "下"]
PREV_MAP = dict(PREV_PATTERN_CHOICES)

DEFAULTS = {
    "prev_label": "不知道",
    "strategy": "kelly",
    "buy_price": None,
    "bought_qty": None,
    **{f"p_{i}": None for i in range(N_SLOTS)},
    **{f"s_{i}": None for i in range(N_SLOTS)},
}

st.set_page_config(page_title="大頭菜買賣顧問", page_icon="🥬", layout="wide")


# --------------------------------------------------------------------------
# 狀態持久化
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
    try:
        saved = json.loads(_ls_get("turnip_state") or "{}")
    except Exception:
        saved = {}
    for k, dv in DEFAULTS.items():
        ss[k] = saved.get(k, dv)
    try:
        ss["budget"] = int(_ls_get("turnip_budget") or 1_000_000)
    except Exception:
        ss["budget"] = 1_000_000
    ss["_init"] = True


def persist_state():
    ss = st.session_state
    _ls_set("turnip_state", json.dumps({k: ss[k] for k in DEFAULTS}))
    _ls_set("turnip_budget", str(int(ss["budget"])))


def clear_data():
    ss = st.session_state
    for k, dv in DEFAULTS.items():
        ss[k] = dv     # 預算 (budget) 不動


@st.cache_resource(show_spinner=False)
def load_advisor():
    return get_advisor()


# --------------------------------------------------------------------------
# 樣式 (海島晨光 / Nook 風) — 注意: 注入的 HTML 內絕不能有空行, 否則 Streamlit 會把後面當純文字
# --------------------------------------------------------------------------
def inject_css():
    css = (
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;700&display=swap');"
        ":root{--cream:#fbf4df;--cream2:#f3e8c7;--leaf:#5fa85f;--leaf-d:#3f7a3f;--brown:#5b4636;--brown-l:#8a7059;--gold:#f4c95d;--shadow:rgba(95,168,95,.18);}"
        "html,body,[class*='css'],.stApp{font-family:'Noto Sans TC','Fredoka',sans-serif;color:var(--brown);}"
        ".stApp{background:radial-gradient(1200px 480px at 50% -8%,#fffaf0 0%,transparent 60%),radial-gradient(900px 500px at 110% 10%,#e8f3da 0%,transparent 55%),radial-gradient(700px 600px at -10% 30%,#e6f1f7 0%,transparent 50%),var(--cream);}"
        ".block-container{padding-top:1.6rem;max-width:1100px;}"
        "#MainMenu,footer,header,[data-testid='stToolbar'],[data-testid='stDecoration'],[data-testid='stStatusWidget']{display:none!important;}"
        "h1,h2,h3,h4{font-family:'Fredoka','Noto Sans TC',sans-serif!important;color:var(--brown)!important;}"
        ".hero{text-align:center;margin-bottom:.3rem;}"
        ".hero .title{font-family:'Fredoka',sans-serif;font-weight:700;font-size:2.4rem;color:var(--leaf-d);}"
        "[data-testid='stVerticalBlockBorderWrapper']{background:rgba(255,255,255,.72);border:1.5px solid #ece0bf!important;border-radius:20px;box-shadow:0 8px 22px var(--shadow);}"
        "[data-testid='stVerticalBlockBorderWrapper'] [data-testid='stVerticalBlockBorderWrapper']{background:#fffdf6;border:1.5px solid #e6d9b6!important;border-radius:14px;box-shadow:none;margin-bottom:.45rem;}"
        ".daytag{font-family:'Fredoka','Noto Sans TC';font-weight:700;font-size:1.05rem;color:var(--leaf-d);border-bottom:1.5px dashed #e6d9b6;padding-bottom:.15rem;margin-bottom:.1rem;}"
        ".colhead{text-align:center;font-weight:700;font-size:.82rem;color:var(--brown-l);padding-bottom:.1rem;}"
        ".hint{background:#f6efd8;border:1.5px solid #e6d9b6;border-radius:12px;padding:.6rem .9rem;color:#8a7059;font-size:.92rem;text-align:center;}"
        ".warn{background:#fbe9cf;border:1.5px solid #f0cfa0;border-radius:12px;padding:.45rem .85rem;color:#9a6a2e;font-size:.9rem;}"
        ".errbox{background:#f6d7de;border:1.5px solid #e8a9b6;border-radius:12px;padding:.6rem .9rem;color:#8f3a4a;font-weight:600;}"
        ".tol{background:#e6f1f7;border:1.5px solid #aacfe2;border-radius:12px;padding:.55rem .9rem;color:#2f5d75;font-weight:600;margin-bottom:.5rem;}"
        "div[role='radiogroup']{gap:.5rem;flex-wrap:wrap;}"
        "div[role='radiogroup'] label{background:#fff;border:2px solid #e3d7b6;border-radius:16px;padding:.5rem 1.25rem!important;margin:0!important;font-size:1.02rem;font-weight:600;transition:.15s;cursor:pointer;}"
        "div[role='radiogroup'] label:hover{border-color:var(--leaf);transform:translateY(-1px);}"
        "div[role='radiogroup'] label:has(input:checked){background:var(--leaf);border-color:var(--leaf-d);color:#fff;box-shadow:0 4px 12px var(--shadow);}"
        "div[role='radiogroup'] label>div:first-child{display:none!important;}"
        ".stNumberInput input{border-radius:10px!important;font-family:'Fredoka',sans-serif;}"
        ".stNumberInput button,[data-testid='stNumberInputStepUp'],[data-testid='stNumberInputStepDown']{display:none!important;}"
        ".stButton>button{border-radius:14px;border:1.5px solid var(--leaf);background:var(--leaf);color:#fff;font-weight:600;}"
        ".stButton>button:hover{background:var(--leaf-d);border-color:var(--leaf-d);color:#fff;}"
        ".big-num{font-family:'Fredoka',sans-serif;font-weight:700;font-size:2rem;color:var(--leaf-d);line-height:1.1;}"
        ".sub{color:var(--brown-l);font-size:.9rem;}"
        ".sgrid{font-size:.85rem;}"
        ".ptable{width:100%;border-collapse:separate;border-spacing:0 4px;font-size:.82rem;}"
        ".ptable th{color:var(--brown-l);font-weight:700;padding:.25rem .3rem;text-align:center;}"
        ".ptable td{text-align:center;padding:.3rem .25rem;background:#fff;white-space:nowrap;}"
        ".ptable td.name{text-align:left;font-weight:700;border-radius:9px 0 0 9px;padding-left:.6rem;background:#faf3df;}"
        ".ptable td.prob{font-weight:700;background:#faf3df;}"
        ".ptable .allrow td{background:#eef6e6;font-weight:600;}"
        ".tablescroll{overflow-x:auto;-webkit-overflow-scrolling:touch;}"
        ".tablescroll .ptable{min-width:560px;}"
        # ---- 手機 / 窄螢幕 (<=640px): 頂層欄位改為直向堆疊, 欄內小列維持橫向 ----
        "@media (max-width:640px){"
        ".block-container{padding-left:.55rem;padding-right:.55rem;padding-top:1rem;max-width:100%;}"
        ".hero .title{font-size:1.7rem;}"
        # 頂層欄位(設定/星期日/日列) 直向堆疊, 且每欄撐滿整列寬
        "[data-testid='stHorizontalBlock']{flex-direction:column;gap:.4rem;}"
        "[data-testid='stColumn']{width:100%!important;flex:1 1 100%!important;min-width:0!important;}"
        # 欄內的時段小列 維持橫向不換行, 內欄平分一列寬
        "[data-testid='stColumn'] [data-testid='stHorizontalBlock']{flex-direction:row;flex-wrap:nowrap;gap:.25rem;}"
        "[data-testid='stColumn'] [data-testid='stColumn']{width:auto!important;flex:1 1 0!important;}"
        ".big-num{font-size:1.6rem;}"
        ".sgrid{font-size:.76rem;}"
        ".stNumberInput input{padding:.3rem .4rem!important;}"
        "div[role='radiogroup'] label{padding:.4rem .9rem!important;font-size:.95rem;}"
        "}"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def fmt(n):
    return f"{int(n):,}"


def cell_style(cmin, cmax, buy, is_obs):
    """依價格區間給底色 (參考 ac-turnip: 已填=灰, 高=綠, 中=藍, 低/賠=粉)。"""
    if is_obs:
        return "background:#c6cbd1;color:#36404a;font-weight:700", ""   # 已填 (灰)
    if buy in (0, None):
        return "background:#f3eede;color:#9a8f78", ""
    rmax = cmax / buy
    if cmax < buy:
        bg, fg = "#f0bcc8", "#8f3a4a"        # 賠錢 (粉)
    elif cmin >= buy or rmax >= 1.35:
        bg, fg = "#a7da82", "#2f6b1f"        # 高 (綠)
    else:
        bg, fg = "#9cc1e6", "#274b73"        # 中 (藍)
    return f"background:{bg};color:{fg};font-weight:700", ""


def render_pattern_table(ptab, buy):
    obs = ptab["obs"]
    h = ["<table class='ptable'><tr><th rowspan='2' style='text-align:left;padding-left:.6rem'>波型</th><th rowspan='2'>機率</th>"]
    for d in DAY_ZH:
        h.append(f"<th colspan='2'>{d}</th>")
    h.append("</tr><tr>")
    for _ in range(6):
        h.append("<th>上</th><th>下</th>")
    h.append("</tr>")

    def cells(ranges):
        out = []
        for i, (mn, mx) in enumerate(ranges):
            is_obs = obs[i] is not None
            sty, extra = cell_style(mn, mx, buy, is_obs)
            txt = f"{mn}" if mn == mx else f"{mn}~{mx}"
            rad = "border-radius:9px" if i == N_SLOTS - 1 else ""
            out.append(f"<td style='{sty};{extra}{rad}'>{txt}</td>")
        return "".join(out)

    rows = ["<div class='tablescroll'>", "".join(h)]
    rows.append(f"<tr class='allrow'><td class='name'>所有波型</td><td class='prob'>—</td>{cells(ptab['overall'])}</tr>")
    for r in ptab["rows"]:
        rows.append(f"<tr><td class='name'>{r['name']}</td><td class='prob'>{r['prob']*100:.0f}%</td>{cells(r['cells'])}</tr>")
    rows.append("</table></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_chart(res, buy):
    t = res["table"]
    full = [f"{DAY_ZH[i//2]} {AMPM[i % 2]}" for i in range(N_SLOTS)]
    smax = [r["smax"] for r in t]
    q90 = [r["q90"] for r in t]
    q10 = [r["q10"] for r in t]
    gmin = [r["gmin"] for r in t]
    obs = [r["obs"] for r in t]
    floor = int(max(gmin)) if gmin else buy   # 保底價格 = 全週最佳保證下限

    C_BUY, C_FLOOR = "#8a6d4f", "#2f9c8c"
    C_MAX, C_LIKELY, C_MIN, C_OBS = "#7bc467", "#7eaad6", "#e896b4", "#e8743b"
    fig = go.Figure()
    # 加入順序 = hover 由上而下順序; zorder 控制視覺疊放
    fig.add_trace(go.Scatter(x=full, y=[buy] * N_SLOTS, mode="lines", name="購買價格",
                  line=dict(color=C_BUY, width=2, dash="dot"), zorder=10,
                  hovertemplate="購買價格：%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=full, y=[floor] * N_SLOTS, mode="lines", name="保底價格",
                  line=dict(color=C_FLOOR, width=2, dash="dot"), zorder=10,
                  hovertemplate="保底價格：%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=full, y=smax, fill="tozeroy", mode="lines", name="最高價格",
                  line=dict(color=C_MAX, width=0), fillcolor="rgba(123,196,103,.5)", zorder=1,
                  hovertemplate="最高價格：%{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=full, y=q90, fill="tozeroy", mode="lines", name="最可能價格",
                  line=dict(color=C_LIKELY, width=0), fillcolor="rgba(126,170,214,.65)", zorder=2,
                  customdata=[f"{q10[i]}-{q90[i]}" for i in range(N_SLOTS)],
                  hovertemplate="最可能價格：%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=full, y=gmin, fill="tozeroy", mode="lines", name="最低價格",
                  line=dict(color=C_MIN, width=0), fillcolor="rgba(232,150,180,.7)", zorder=3,
                  hovertemplate="最低價格：%{y:.0f}<extra></extra>"))
    ox = [full[i] for i in range(N_SLOTS) if obs[i] is not None]
    oy = [obs[i] for i in range(N_SLOTS) if obs[i] is not None]
    if ox:
        fig.add_trace(go.Scatter(x=ox, y=oy, mode="markers", name="你的價格",
                      marker=dict(size=10, color=C_OBS, line=dict(width=1.5, color="#fff")),
                      zorder=20, hovertemplate="你的價格：%{y:.0f}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=8, r=8, t=8, b=8),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.16, x=0),
                      hovermode="x unified", hoverlabel=dict(font=dict(family="Noto Sans TC")),
                      font=dict(family="Noto Sans TC", color="#5b4636"))
    fig.update_yaxes(rangemode="tozero", gridcolor="rgba(0,0,0,.06)")
    fig.update_xaxes(showgrid=False, tickangle=0, tickfont=dict(size=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# --------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------
def main():
    inject_css()
    init_state()
    ss = st.session_state
    advisor = load_advisor()

    st.markdown("<div class='hero'><div class='title'>🥬 大頭菜買賣顧問</div></div>",
                unsafe_allow_html=True)

    # ===== 設定: 上週波型 / 策略 =====
    with st.container(border=True):
        st.markdown("#### 上週波型")
        st.radio("上週波型", [c[0] for c in PREV_PATTERN_CHOICES], key="prev_label",
                 horizontal=True, label_visibility="collapsed")
        st.markdown("#### 策略風格")
        st.radio("策略", list(STRATEGIES), key="strategy", horizontal=True,
                 format_func=lambda s: STRATEGY_INFO[s]["label"], label_visibility="collapsed")
        st.markdown("\n".join(
            f"- **{STRATEGY_INFO[s]['label']}**：{STRATEGY_INFO[s]['desc']}" for s in STRATEGIES))

    prev_pattern = PREV_MAP[ss["prev_label"]]
    strategy = ss["strategy"]
    budget = int(ss["budget"])

    # ===== 星期日: 買入 =====
    with st.container(border=True):
        st.markdown("#### ☀️ 星期日 · 買入")
        c1, c2, c3, c4 = st.columns([1, 1, 1.1, 1])
        c1.number_input("預算（鈴錢）", min_value=0, key="budget")
        c2.number_input("買入價格", min_value=0, max_value=110, key="buy_price")
        budget = int(ss["budget"] or 0)
        buy_price = int(ss["buy_price"] or 0)
        if buy_price and buy_price < 90:
            st.markdown("<div class='warn'>買價通常是 90~110</div>", unsafe_allow_html=True)
        if buy_price >= 90:
            res0 = advise(advisor, buy_price, prev_pattern, observed=[], budget=budget, holding=0)
            b = res0["buy"][strategy]
            c3.markdown(f"<div class='sub'>建議買入</div><div class='big-num'>{fmt(b['qty'])} 顆</div>"
                        f"<div class='sub'>約 {fmt(b['bells'])} 鈴 · {b['tag']}</div>",
                        unsafe_allow_html=True)
            c4.number_input("實際買入顆數", min_value=0, key="bought_qty")
        else:
            c3.markdown("<div class='hint'>輸入買價後顯示建議</div>", unsafe_allow_html=True)

    buy_price = int(ss["buy_price"] or 0)
    bought_qty = round_down_10(ss["bought_qty"] or 0)

    # 讀週間價格 + 算每格持有量 (未填 = None, 由其他價格推估補上)
    prices = [(int(ss[f"p_{i}"]) if ss[f"p_{i}"] else None) for i in range(N_SLOTS)]
    known = [i for i, v in enumerate(prices) if v is not None]
    cur_slot = max(known) if known else None
    holding_at, h = [0] * N_SLOTS, bought_qty
    for i in range(N_SLOTS):
        holding_at[i] = h
        h -= min(round_down_10(ss[f"s_{i}"] or 0), h)
    remaining = h

    # ===== 週間: 賣出 (雙欄, 週一~三 / 週四~六) =====
    with st.container(border=True):
        head = st.columns([1.4, 1])
        head[0].markdown("#### 🛒 週一～週六 · 賣出")
        head[1].markdown(f"<div style='text-align:right' class='sub'>目前持有<br>"
                         f"<span class='big-num' style='font-size:1.4rem'>{fmt(max(remaining,0))}</span> 顆</div>",
                         unsafe_allow_html=True)
        def day_box(d):
            """單日(上午/下午)輸入框。欄位用 placeholder 自我標示, 不另設表頭。"""
            with st.container(border=True):
                st.markdown(f"<div class='daytag'>{DAY_ZH[d]}</div>", unsafe_allow_html=True)
                for ap in range(2):
                    i = d * 2 + ap
                    sugg_q, sugg_t = (0, "—")
                    if buy_price >= 90 and prices[i] is not None and holding_at[i] > 0:
                        sugg_q, sugg_t = slot_sell_qty(advisor, buy_price, prev_pattern,
                                                       prices, i, holding_at[i], strategy)
                    r = st.columns([0.7, 1, 1.2, 1])
                    mark = " 🟢" if i == cur_slot else ""
                    r[0].markdown(f"<div style='padding-top:.55rem;font-weight:600'>{AMPM[ap]}{mark}</div>",
                                  unsafe_allow_html=True)
                    r[1].number_input(f"price_{i}", min_value=0, max_value=999, placeholder="菜價",
                                      key=f"p_{i}", label_visibility="collapsed")
                    if prices[i] is not None and bought_qty:
                        sugg_html = f"建議賣 <b>{fmt(sugg_q)}</b> 顆<br><span class='sub'>{sugg_t}</span>"
                    else:
                        sugg_html = "<span class='sub'>建議賣出</span>"
                    r[2].markdown(f"<div style='padding-top:.4rem' class='sgrid'>{sugg_html}</div>",
                                  unsafe_allow_html=True)
                    r[3].number_input(f"sold_{i}", min_value=0, placeholder="實際賣",
                                      key=f"s_{i}", label_visibility="collapsed")

        # row-base 排列: (一,二) / (三,四) / (五,六)
        for rowi in range(3):
            rc = st.columns(2)
            for ci in range(2):
                with rc[ci]:
                    day_box(rowi * 2 + ci)

    # ===== 預測: 圖表 + 波型×時間 表 =====
    if buy_price >= 90:
        observed = prices[:cur_slot + 1] if cur_slot is not None else []
        res = advise(advisor, buy_price, prev_pattern, observed=observed, budget=budget, holding=0)
        ptab = pattern_table(buy_price, prev_pattern, observed)
        with st.container(border=True):
            st.markdown("#### 🔮 價格預測")
            if not res["feasible"] or not ptab.get("feasible"):
                st.markdown("<div class='errbox'>這組價格在官方規則下不可能同時出現<br>"
                            "<span style='font-weight:400;font-size:.9rem'>請確認「買入價格」與各時段價格是否正確"
                            "（常見：買價記錯一兩元，或同一天兩格不該相等）</span></div>",
                            unsafe_allow_html=True)
            else:
                tol = ptab.get("tolerance_used", 0)
                if tol > 0:
                    best = max(ptab["rows"], key=lambda r: r["prob"]) if ptab["rows"] else None
                    bname = best["name"] if best else "—"
                    st.markdown(
                        f"<div class='tol'>容錯模式：輸入與官方規則差約 ±{tol} 鈴，"
                        f"已用最接近的波型推估（最相符：{bname}）</div>",
                        unsafe_allow_html=True)
                render_chart(res, buy_price)
                render_pattern_table(ptab, buy_price)
                st.markdown("<div class='sub'>顏色：🟩 高　🟦 中　🟥 賠　⬜ 已填</div>",
                            unsafe_allow_html=True)

    # ===== 清除 =====
    cL, cR = st.columns([5, 1])
    cR.button("🧹 清除資料", on_click=clear_data, width="stretch")

    persist_state()


if __name__ == "__main__":
    main()
