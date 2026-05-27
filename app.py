"""
app.py — Option Pricing Dashboard
===================================
Streamlit app: fetch live option chain, price with BS / CRR / Monte Carlo.
Supports both European and American option styles (American via CRR only).

Run locally:
    streamlit run app.py
"""

import io
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

from utils import (
    bs_price,
    bs_implied_vol_single,
    binomial_price,
    early_exercise_premium,
    mc_price,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Option Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"], p, li, span, div {
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.45rem !important;
    color: #58a6ff !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #8b949e !important;
}
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
    border-bottom: 1px solid #30363d;
    padding-bottom: 14px;
    margin-bottom: 8px !important;
}
h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 28px !important;
    margin-bottom: 6px !important;
}
hr {
    border: none !important;
    border-top: 1px solid #21262d !important;
    margin: 28px 0 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
.stButton > button {
    background-color: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 8px 18px !important;
    width: 100% !important;
}
.stButton > button:hover {
    background-color: #2ea043 !important;
    border-color: #3fb950 !important;
}
[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #c9d1d9 !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #8b949e !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stCaptionContainer"] p {
    font-size: 0.8rem !important;
    color: #6e7681 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def badge(label: str, color: str) -> str:
    colors = {
        "green":  ("#1a3a2a", "#3fb950", "#3fb950"),
        "blue":   ("#0d2d5e", "#58a6ff", "#58a6ff"),
        "orange": ("#3d2000", "#f0883e", "#f0883e"),
        "purple": ("#2d1a4a", "#a78bfa", "#a78bfa"),
    }
    bg, fg, border = colors.get(color, colors["blue"])
    return (
        f"<span style='display:inline-block;padding:3px 12px;"
        f"background:{bg};color:{fg};border:1px solid {border};"
        f"border-radius:12px;font-family:IBM Plex Mono,monospace;"
        f"font-size:0.72rem;font-weight:600;letter-spacing:0.05em'>"
        f"{label}</span>"
    )


def info_box(text: str) -> None:
    st.markdown(
        f"<div style='background:#0d2d5e22;border-left:3px solid #58a6ff;"
        f"padding:10px 16px;border-radius:0 6px 6px 0;font-size:0.84rem;"
        f"color:#8b949e;margin:10px 0 18px 0;line-height:1.55'>{text}</div>",
        unsafe_allow_html=True,
    )


def warning_box(text: str) -> None:
    st.markdown(
        f"<div style='background:#3d200022;border-left:3px solid #f0883e;"
        f"padding:10px 16px;border-radius:0 6px 6px 0;font-size:0.84rem;"
        f"color:#8b949e;margin:10px 0 18px 0;line-height:1.55'>{text}</div>",
        unsafe_allow_html=True,
    )


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono", color="#c9d1d9", size=11),
    title_font=dict(size=13, color="#e6edf3"),
    legend=dict(bgcolor="#0d1117", bordercolor="#30363d",
                borderwidth=1, font=dict(size=11)),
    margin=dict(l=48, r=24, t=52, b=40),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
)


# ─────────────────────────────────────────────
# CACHED DATA HELPERS
# ─────────────────────────────────────────────

def _fetch_chain_with_retry(tk, expiry: str, max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            chain = tk.option_chain(expiry)
            calls = chain.calls.copy(); calls["type"] = "call"
            puts  = chain.puts.copy();  puts["type"]  = "put"
            df    = pd.concat([calls, puts], ignore_index=True, sort=False)
            if len(df) > 0:
                return df
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_option_data(ticker: str, n_expiries: int, cache_bust: str):
    tk       = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise ValueError(f"No options found for {ticker}.")

    spot      = float(tk.history(period="1d")["Close"].iloc[-1])
    debug_log = []
    all_dfs   = []

    for e in expiries[:n_expiries]:
        df = _fetch_chain_with_retry(tk, e)
        if len(df) > 0:
            df["expiry"] = pd.to_datetime(e)
            all_dfs.append(df)
            debug_log.append(f"✓  {e}  →  {len(df)} rows")
        else:
            debug_log.append(f"✗  {e}  →  empty (skipped)")
        time.sleep(0.5)

    if not all_dfs:
        raise ValueError(
            "All chains returned empty. Yahoo Finance is rate limiting "
            "this server. Wait 3-5 minutes and try again."
        )

    opts = pd.concat(all_dfs, ignore_index=True, sort=False)

    for col in ["bid", "ask", "lastPrice", "volume", "openInterest"]:
        opts[col] = pd.to_numeric(opts.get(col), errors="coerce")

    opts["mid"]          = opts[["bid", "ask"]].mean(axis=1).fillna(opts["lastPrice"])
    opts["mid"]          = opts["mid"].where(opts["mid"] > 0, opts["lastPrice"])
    opts["mid"]          = opts["mid"].fillna(opts["lastPrice"])
    opts["volume"]       = opts["volume"].fillna(0)
    opts["openInterest"] = opts["openInterest"].fillna(0)

    today          = pd.Timestamp.today().normalize()
    opts["T_days"] = (opts["expiry"] - today).dt.days
    opts           = opts[opts["T_days"] >= 0].copy()
    opts["T"]      = opts["T_days"] / 365.0
    opts["strike"] = pd.to_numeric(opts["strike"], errors="coerce")
    opts["spot"]   = spot
    opts["moneyness"] = opts["strike"] / spot

    opts = opts[
        (opts["mid"]    > 0)    &
        (opts["strike"] > 0)    &
        (opts["T"]      >= 1/365)
    ].reset_index(drop=True)

    return opts, spot, debug_log


@st.cache_data(ttl=300, show_spinner=False)
def compute_ivs(demo_json: str, r: float):
    demo_df = pd.read_json(io.StringIO(demo_json))

    def _row(row):
        iv   = bs_implied_vol_single(
            row["mid"], row["spot"], row["strike"],
            row["T"], r, option_type=row["type"],
        )
        sig  = iv if not np.isnan(iv) else 0.2
        bs_p = bs_price(row["spot"], row["strike"], row["T"], r,
                        sig, option_type=row["type"])
        return pd.Series({"iv_calc": iv, "bs_price": bs_p})

    demo_df[["iv_calc", "bs_price"]] = demo_df.apply(_row, axis=1)
    return demo_df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Config")
    st.divider()

    st.markdown("**Market Data**")
    ticker     = st.text_input("Ticker", value="SPY").upper().strip()
    n_expiries = st.slider("Expiries to fetch", 2, 12, 6)
    fetch_btn  = st.button("↻  Fetch Live Data")

    st.divider()

    st.markdown("**Parameters**")
    r = st.number_input(
        "Risk-free rate (r)",
        min_value=0.0, max_value=0.20,
        value=0.05, step=0.005, format="%.3f",
        help="Annualised rate. Use 3-month T-bill yield for accuracy.",
    )

    st.divider()

    st.markdown("**Manual Pricer**")
    st.caption("Override inputs to price one option directly.")
    manual_spot  = st.number_input("Spot (S)",           value=580.0, step=1.0)
    manual_K     = st.number_input("Strike (K)",         value=580.0, step=1.0)
    manual_T     = st.number_input("Maturity T (years)", value=0.25,
                                   min_value=0.003, step=0.01, format="%.3f")
    manual_sigma = st.number_input("Volatility σ",       value=0.20,
                                   min_value=0.01, max_value=5.0,
                                   step=0.01, format="%.2f")
    manual_type  = st.selectbox("Option type", ["call", "put"])

    # ── Option style selector ────────────────────────────────────────
    st.markdown("**Option Style**")
    option_style = st.radio(
        "Exercise style",
        ["European", "American"],
        horizontal=True,
        help=(
            "European: exercise at expiry only.\n"
            "American: exercise at any time — priced via CRR binomial tree.\n"
            "Note: BS and MC price European options only."
        ),
    )
    option_style_key = option_style.lower()

    st.divider()

    st.markdown("**Model Settings**")
    crr_steps       = st.slider("CRR steps",       50,  500, 100, step=50)
    mc_paths_k      = st.slider("MC paths (000s)", 10,  500, 100, step=10)
    mc_paths_actual = mc_paths_k * 1_000

    st.divider()
    st.markdown(
        "<p style='text-align:center;font-family:IBM Plex Mono,monospace;"
        "font-size:0.68rem;color:#484f58;margin:0'>BS · CRR · Monte Carlo</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────

st.markdown("<h1>📈 Option Pricing Dashboard</h1>", unsafe_allow_html=True)

b1, b2, b3, b4, _ = st.columns([1, 1, 1, 1.2, 2])
b1.markdown(badge("BLACK-SCHOLES", "green"),  unsafe_allow_html=True)
b2.markdown(badge("CRR BINOMIAL",  "blue"),   unsafe_allow_html=True)
b3.markdown(badge("MONTE CARLO",   "orange"), unsafe_allow_html=True)
b4.markdown(badge(f"EXERCISE: {option_style.upper()}", "purple"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — MANUAL SINGLE-OPTION PRICER
# ─────────────────────────────────────────────

st.markdown("### Single Option Pricer")
info_box(
    "Price one option with all three models using the sidebar inputs. "
    "Adjust spot, strike, vol, or maturity and prices update instantly. "
    "Switch between European and American exercise style using the sidebar toggle."
)

# BS — European only
bs_val = bs_price(manual_spot, manual_K, manual_T, r,
                  manual_sigma, option_type=manual_type)

# CRR — supports both styles
bin_val = binomial_price(manual_spot, manual_K, manual_T, r,
                         manual_sigma, steps=crr_steps,
                         option_type=manual_type,
                         option_style=option_style_key)

# MC — European only
mc_val, mc_se = mc_price(manual_spot, manual_K, manual_T, r,
                          manual_sigma, n_paths=mc_paths_actual,
                          option_type=manual_type, seed=42)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Black-Scholes (EUR)", f"${bs_val:.4f}"  if not np.isnan(bs_val)  else "—")
col2.metric(f"CRR ({option_style[:3].upper()})",
            f"${bin_val:.4f}" if not np.isnan(bin_val) else "—")
col3.metric("Monte Carlo (EUR)",   f"${mc_val:.4f}"  if not np.isnan(mc_val)  else "—")
col4.metric("MC Std Error",        f"${mc_se:.5f}"   if not np.isnan(mc_se)   else "—")

# Diffs
if not any(np.isnan(v) for v in [bs_val, bin_val, mc_val]):
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2, *_ = st.columns(4)
    d1.metric("|BS − CRR|", f"${abs(bs_val - bin_val):.5f}")
    d2.metric("|BS − MC|",  f"${abs(bs_val - mc_val):.5f}")

# American early exercise premium display
if option_style_key == "american":
    am_val, eur_val, prem = early_exercise_premium(
        manual_spot, manual_K, manual_T, r,
        manual_sigma, steps=crr_steps, option_type=manual_type
    )
    if not np.isnan(prem):
        st.markdown("<br>", unsafe_allow_html=True)
        ep1, ep2, ep3 = st.columns(3)
        ep1.metric("CRR American",         f"${am_val:.4f}")
        ep2.metric("CRR European",         f"${eur_val:.4f}")
        ep3.metric("Early Exercise Premium", f"${prem:.5f}",
                   help="American − European. Always ≥ 0. "
                        "Non-zero primarily for deep ITM puts.")
        if manual_type == "call" and prem < 0.001:
            warning_box(
                "Early exercise premium ≈ 0 for this call — consistent with theory. "
                "On a non-dividend paying asset it is never optimal to exercise a call early."
            )
        elif prem > 0.01:
            info_box(
                f"Early exercise premium of ${prem:.4f} indicates that at some nodes during "
                f"backward induction, immediate exercise dominates holding. "
                f"This is most common for deep ITM puts where interest on the strike "
                f"exceeds the remaining time value."
            )

st.divider()


# ─────────────────────────────────────────────
# SECTION 2 — LIVE DATA
# ─────────────────────────────────────────────

st.markdown("### Live Market Data")

if "opts" not in st.session_state:
    st.session_state.opts      = None
    st.session_state.spot      = None
    st.session_state.debug_log = []

if fetch_btn or st.session_state.opts is None:
    with st.spinner(f"Fetching {ticker} option chain — this may take 30-60s..."):
        try:
            bust = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            opts, spot, debug_log      = fetch_option_data(ticker, n_expiries, cache_bust=bust)
            st.session_state.opts      = opts
            st.session_state.spot      = spot
            st.session_state.debug_log = debug_log
            st.success(
                f"✓  {ticker} — {len(opts):,} contracts loaded  "
                f"|  Spot: **${spot:.2f}**"
            )
        except Exception as exc:
            st.error(f"⚠️ {exc}")
            if st.session_state.debug_log:
                with st.expander("Debug — per-expiry fetch log"):
                    st.code("\n".join(st.session_state.debug_log))

opts = st.session_state.opts
spot = st.session_state.spot

if opts is None:
    st.info("👈  Click **↻ Fetch Live Data** in the sidebar to load options.")
    st.stop()

if st.session_state.debug_log:
    with st.expander("Debug — per-expiry fetch log", expanded=False):
        st.code("\n".join(st.session_state.debug_log))

# Expiry selector
expiries    = sorted(opts["expiry"].unique())
expiry_strs = [str(e.date()) for e in expiries]
sel_str     = st.selectbox("Select expiry", expiry_strs)

if not sel_str:
    st.info("Select an expiry above to continue.")
    st.stop()

demo_expiry = expiries[expiry_strs.index(sel_str)]
demo_raw    = opts[opts["expiry"] == demo_expiry].copy().reset_index(drop=True)

# Compute IVs
with st.spinner("Computing implied volatilities..."):
    demo_df = compute_ivs(demo_raw.to_json(), r)

# Apply CRR (European) + MC
with st.spinner("Pricing with CRR & Monte Carlo..."):

    def _bin_eur(row):
        if pd.isna(row["iv_calc"]):
            return np.nan
        return binomial_price(
            row["spot"], row["strike"], row["T"], r,
            row["iv_calc"], steps=crr_steps,
            option_type=row["type"], option_style="european",
        )

    def _mc(row):
        if pd.isna(row["iv_calc"]) or row["T"] <= 0:
            return pd.Series({"mc_price": np.nan, "mc_se": np.nan})
        p, se = mc_price(
            row["spot"], row["strike"], row["T"], r,
            row["iv_calc"], n_paths=mc_paths_actual,
            option_type=row["type"], seed=None,
        )
        return pd.Series({"mc_price": p, "mc_se": se})

    demo_df["binomial_eur"]           = demo_df.apply(_bin_eur, axis=1)
    demo_df[["mc_price", "mc_se"]]    = demo_df.apply(_mc,      axis=1)
    demo_df["bs_error"]               = demo_df["bs_price"]    - demo_df["mid"]
    demo_df["bin_error"]              = demo_df["binomial_eur"] - demo_df["mid"]
    demo_df["mc_error"]               = demo_df["mc_price"]     - demo_df["mid"]

# Apply CRR American + early exercise premium
with st.spinner("Computing American prices & early exercise premiums..."):

    def _early_ex(row):
        if pd.isna(row["iv_calc"]) or row["T"] <= 0:
            return pd.Series({"binomial_am": np.nan, "ee_premium": np.nan})
        am, eu, prem = early_exercise_premium(
            row["spot"], row["strike"], row["T"], r,
            row["iv_calc"], steps=crr_steps, option_type=row["type"],
        )
        return pd.Series({"binomial_am": am, "ee_premium": prem})

    demo_df[["binomial_am", "ee_premium"]] = demo_df.apply(_early_ex, axis=1)

st.divider()


# ─────────────────────────────────────────────
# SECTION 3 — THREE-MODEL COMPARISON TABLE
# ─────────────────────────────────────────────

st.markdown("### Three-Model Comparison")
info_box(
    "All models priced using market-implied vol as σ. "
    "BS and MC price European options only. "
    "CRR prices both styles — Early Exercise Premium = American − European."
)

mae = lambda x: float(np.nanmean(np.abs(x)))
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE — Black-Scholes",   f"${mae(demo_df['bs_error']):.5f}")
c2.metric("MAE — CRR (European)",  f"${mae(demo_df['bin_error']):.5f}")
c3.metric("MAE — Monte Carlo",     f"${mae(demo_df['mc_error']):.5f}")
c4.metric("Avg EE Premium",
          f"${demo_df['ee_premium'].mean():.5f}"
          if demo_df['ee_premium'].notna().any() else "—")

st.markdown("<br>", unsafe_allow_html=True)

display_cols = ["strike", "type", "mid", "bs_price", "binomial_eur",
                "binomial_am", "ee_premium", "mc_price",
                "bs_error", "bin_error", "mc_error", "iv_calc"]
available    = [c for c in display_cols if c in demo_df.columns]
table_df     = demo_df[available].dropna(subset=["bs_price"]).copy()

for col in available:
    if col not in ["strike", "type"]:
        table_df[col] = table_df[col].round(4)

st.dataframe(
    table_df.rename(columns={
        "strike": "Strike", "type": "Type",
        "mid": "Mid (Mkt)", "bs_price": "BS (EUR)",
        "binomial_eur": "CRR (EUR)", "binomial_am": "CRR (AM)",
        "ee_premium": "EE Premium", "mc_price": "MC (EUR)",
        "bs_error": "BS Err", "bin_error": "CRR Err",
        "mc_error": "MC Err", "iv_calc": "Impl. Vol",
    }),
    width="stretch",
    height=320,
)

st.divider()


# ─────────────────────────────────────────────
# SECTION 4 — EARLY EXERCISE PREMIUM
# ─────────────────────────────────────────────

st.markdown("### Early Exercise Premium")
info_box(
    "American option price minus European option price (CRR tree). "
    "Always ≥ 0 by no-arbitrage. For calls on non-dividend paying assets this is ~0. "
    "Premium is largest for deep ITM puts where interest on the strike exceeds remaining time value."
)

prem_df = demo_df.dropna(subset=["ee_premium"]).copy()

# Split calls vs puts for annotation
calls_prem = prem_df[prem_df["type"] == "call"]["ee_premium"]
puts_prem  = prem_df[prem_df["type"] == "put"]["ee_premium"]

p1, p2, p3, p4 = st.columns(4)
p1.metric("Calls with EE Premium > 0",    f"{(calls_prem > 0.001).sum()}")
p2.metric("Puts with EE Premium > 0",     f"{(puts_prem  > 0.001).sum()}")
p3.metric("Max Put Premium",              f"${puts_prem.max():.4f}" if len(puts_prem) > 0 else "—")
p4.metric("Max Call Premium",             f"${calls_prem.max():.4f}" if len(calls_prem) > 0 else "—")

fig_prem = px.scatter(
    prem_df,
    x="strike", y="ee_premium",
    color="type",
    color_discrete_map={"call": "#58a6ff", "put": "#f0883e"},
    size="mid", size_max=14, opacity=0.80,
    labels={"ee_premium": "EE Premium (AM - EUR, $)",
            "strike": "Strike", "type": "Type"},
)
fig_prem.add_hline(y=0, line_dash="dash", line_color="#484f58", line_width=1.5)
fig_prem.update_layout(**PLOTLY_LAYOUT)
fig_prem.update_layout(title=f"Early Exercise Premium by Strike  ({sel_str})")
st.plotly_chart(fig_prem, width="stretch")

st.divider()


# ─────────────────────────────────────────────
# SECTION 5 — ERROR VS STRIKE
# ─────────────────────────────────────────────

st.markdown("### Pricing Error vs Strike")
info_box(
    "Model price minus market mid across the full strike range. "
    "Near-zero errors near ATM are expected when pricing with market IV."
)

error_model = st.radio(
    "Select model",
    ["Black-Scholes", "CRR (European)", "CRR (American)", "Monte Carlo"],
    horizontal=True,
)
error_col = {
    "Black-Scholes":    "bs_error",
    "CRR (European)":   "bin_error",
    "CRR (American)":   None,
    "Monte Carlo":      "mc_error",
}[error_model]

if error_col is None:
    # American error vs market
    demo_df["am_error"] = demo_df["binomial_am"] - demo_df["mid"]
    scatter_df = demo_df.dropna(subset=["am_error"]).copy()
    fig_err = px.scatter(
        scatter_df, x="strike", y="am_error",
        color="type",
        color_discrete_map={"call": "#58a6ff", "put": "#f0883e"},
        size="mid", size_max=14, opacity=0.80,
        labels={"am_error": "Error (model - mid)", "strike": "Strike", "type": "Type"},
    )
else:
    scatter_df = demo_df.dropna(subset=[error_col]).copy()
    fig_err = px.scatter(
        scatter_df, x="strike", y=error_col,
        color="type",
        color_discrete_map={"call": "#58a6ff", "put": "#f0883e"},
        size="mid", size_max=14, opacity=0.80,
        labels={error_col: "Error (model - mid)", "strike": "Strike", "type": "Type"},
    )

fig_err.add_hline(y=0, line_dash="dash", line_color="#484f58", line_width=1.5)
fig_err.update_layout(**PLOTLY_LAYOUT)
fig_err.update_layout(title=f"{error_model} — Error vs Strike  ({sel_str})")
st.plotly_chart(fig_err, width="stretch")

st.divider()


# ─────────────────────────────────────────────
# SECTION 6 — MC CONVERGENCE
# ─────────────────────────────────────────────

st.markdown("### Monte Carlo Convergence")
info_box(
    "How |MC - BS| and standard error decay as path count increases. "
    "Computed on the nearest ATM contract for the selected expiry."
)

conv_df = demo_df.dropna(subset=["iv_calc"]).copy()
conv_df["atm_dist"] = abs(conv_df["strike"] - spot)
conv_row = conv_df.sort_values("atm_dist").iloc[0]

st.caption(
    f"Contract: **{conv_row['type'].upper()}**  ·  "
    f"Strike: **{conv_row['strike']:.1f}**  ·  "
    f"IV: **{conv_row['iv_calc']:.3f}**  ·  "
    f"T: **{conv_row['T']:.3f}y**  ·  *(nearest ATM)*"
)

paths_list = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 500_000]
bs_ref = bs_price(
    conv_row["spot"], conv_row["strike"], conv_row["T"], r,
    conv_row["iv_calc"], option_type=conv_row["type"],
)

with st.spinner("Running convergence study..."):
    errs, ses = [], []
    for n in paths_list:
        p, se = mc_price(
            conv_row["spot"], conv_row["strike"], conv_row["T"], r,
            conv_row["iv_calc"], n_paths=n,
            option_type=conv_row["type"], seed=11,
        )
        errs.append(abs(p - bs_ref))
        ses.append(se)

fig_conv = go.Figure()
fig_conv.add_trace(go.Scatter(
    x=paths_list, y=errs, mode="lines+markers", name="|MC - BS|",
    line=dict(color="#f0883e", width=2.5), marker=dict(size=8, symbol="circle"),
))
fig_conv.add_trace(go.Scatter(
    x=paths_list, y=ses, mode="lines+markers", name="Std Error",
    line=dict(color="#58a6ff", width=2.5, dash="dot"),
    marker=dict(size=8, symbol="diamond"),
))
fig_conv.update_layout(**PLOTLY_LAYOUT)
fig_conv.update_layout(
    title="MC Convergence — Paths vs Error",
    xaxis_title="Number of Paths",
    yaxis_title="Error ($)",
)
st.plotly_chart(fig_conv, width="stretch")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center;font-family:IBM Plex Mono,monospace;"
    "font-size:0.7rem;color:#484f58;margin:4px 0 20px 0'>"
    "Live data via yfinance &nbsp;·&nbsp; "
    "Models: Black-Scholes · CRR Binomial (European & American) · Monte Carlo GBM<br>"
    "For educational purposes only — not financial advice."
    "</p>",
    unsafe_allow_html=True,
)
