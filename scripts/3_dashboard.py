# scripts/3_dashboard.py
"""
Interactive UAE Stock Market Dashboard (Streamlit + Plotly)
Replaces the previous app; robust to missing selections, recomputes metrics on-the-fly,
separates stocks vs benchmarks, and uses interactive Plotly charts.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ---------- Config ----------
DATA_CSV = Path(__file__).parents[1] / "data" / "uae_stock_data.csv"
st.set_page_config(layout="wide", page_title="UAE Stock Market Dashboard (Interactive)")

# ---------- Helpers ----------
def load_data(path: Path):
    if not path.exists():
        st.error(f"Data file not found: {path}. Run the prepare script first.")
        st.stop()
    # try parse with dayfirst, fallback to default
    df = pd.read_csv(path, index_col='Date')
    try:
        df.index = pd.to_datetime(df.index, dayfirst=True)
    except Exception:
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()
    # coerce columns to numeric when possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')
    return df

def detect_benchmarks(columns):
    # heuristic: common substrings for benchmarks (case-insensitive)
    bench_keywords = ['DFM', 'ADX', 'INDEX', 'DFMGI',
                      'GOLD', 'CRUDE', 'WTI', 'BRENT', 'OIL', 'NATURAL', 'GAS', 'NATGAS', 'BRN']
    benches = []
    for c in columns:
        cu = c.upper()
        if any(k in cu for k in bench_keywords):
            benches.append(c)
    return benches

def clamp_date_range(start, end, min_date, max_date):
    # ensure start <= end and within min/max
    if start is None:
        start = min_date
    if end is None:
        end = max_date
    if start > end:
        start, end = end, start
    start = max(start, min_date)
    end = min(end, max_date)
    return start, end

def annualize_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def annualize_return(daily_returns):
    # geometric annualized
    return (1 + daily_returns.mean())**252 - 1

def calculate_metrics(price_df, rf_rate=0.031, market_series=None):
    """
    price_df: DataFrame of prices (Date index, tickers as columns)
    rf_rate: annual risk-free rate (decimal)
    market_series: Series of market prices if beta required (same index)
    Returns: DataFrame with ROI_%, Annual_Return_%, Annual_Volatility_%, Sharpe, Beta_vs_Market (if available)
    """
    results = []
    returns = price_df.pct_change().dropna()
    for col in price_df.columns:
        ser = price_df[col].dropna()
        if ser.empty:
            results.append({"Ticker": col, "ROI_%": np.nan, "Annual_Return_%": np.nan,
                            "Annual_Volatility_%": np.nan, "Sharpe": np.nan, "Beta_vs_Market": np.nan})
            continue
        roi = (ser.iloc[-1] / ser.iloc[0] - 1) * 100
        if col in returns.columns:
            r = returns[col].dropna()
            ann_vol = annualize_volatility(r)
            ann_ret = annualize_return(r)
            sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol != 0 else np.nan
        else:
            ann_vol = np.nan
            ann_ret = np.nan
            sharpe = np.nan

        beta = np.nan
        if market_series is not None and col in returns.columns:
            # align
            mret = market_series.pct_change().dropna()
            common_idx = returns[col].index.intersection(mret.index)
            if not common_idx.empty and mret.loc[common_idx].var() != 0:
                cov = returns[col].loc[common_idx].cov(mret.loc[common_idx])
                beta = cov / mret.loc[common_idx].var()

        results.append({
            "Ticker": col,
            "ROI_%": round(roi, 3),
            "Annual_Return_%": round(float(ann_ret*100) if not np.isnan(ann_ret) else np.nan, 3),
            "Annual_Volatility_%": round(float(ann_vol*100) if not np.isnan(ann_vol) else np.nan, 3),
            "Sharpe": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
            "Beta_vs_Market": round(beta, 3) if not np.isnan(beta) else np.nan
        })
    return pd.DataFrame(results).set_index("Ticker")

# ---------- App ----------
st.title("UAE Stock Market Analysis — Interactive")

# load
with st.spinner("Loading price data..."):
    df = load_data(DATA_CSV)

if df.empty:
    st.error("Price data is empty after loading. Check your CSV.")
    st.stop()

# detect benches
auto_benchmarks = detect_benchmarks(df.columns)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    # risk-free
    rf_rate = st.number_input("Risk-free rate (annual %)", value=3.1, min_value=0.0, step=0.1) / 100.0

    # date range
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    # coerce to date pair
    if isinstance(date_range, tuple) or isinstance(date_range, list):
        start_date, end_date = date_range[0], date_range[1]
    else:
        # single date selected
        start_date = date_range
        end_date = date_range

    start_date, end_date = clamp_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date),
                                            pd.to_datetime(min_date), pd.to_datetime(max_date))

    # benchmarks - allow user to override detected
    st.markdown("**Benchmarks (auto-detected)**")
    bench_candidates = auto_benchmarks.copy()
    # allow user to manually add/remove known names
    benches_selected_initial = bench_candidates[:1] if bench_candidates else []
    benches_selected = st.multiselect("Choose benchmarks", options=list(df.columns),
                                      default=benches_selected_initial)
    st.markdown("---")
    st.markdown("**Stocks**")
    # stocks options exclude any selected benchmarks
    stocks_options = [c for c in df.columns if c not in benches_selected]
    default_stocks = stocks_options[:3] if len(stocks_options) >= 3 else stocks_options
    stocks_selected = st.multiselect("Choose stocks", options=stocks_options, default=default_stocks)

    st.markdown("---")
    st.markdown("Display")
    show_corr = st.checkbox("Show correlation heatmap (selected)", value=True)
    show_risk_return = st.checkbox("Show risk-return scatter", value=True)
    rolling_days = st.slider("Rolling window for moving averages & volatility (days)", min_value=0, max_value=252, value=50, step=1)

# filter date range
dff_full = df.loc[start_date:end_date]

# --- Panels: Metrics (compute on-the-fly)
# choose market series for beta: prefer first selected benchmark, otherwise none
market_series = None
if benches_selected:
    market_series = dff_full[benches_selected[0]] if benches_selected[0] in dff_full.columns else None

# compute metrics for selected stocks + benchmarks (so user sees both)
combined_selection = list(dict.fromkeys(stocks_selected + benches_selected))  # preserve order, unique
if combined_selection:
    price_slice = dff_full[combined_selection].copy()
    metrics_df = calculate_metrics(price_slice, rf_rate=rf_rate, market_series=market_series)
else:
    metrics_df = pd.DataFrame()  # empty

# ---------- UI: handle empty selection gracefully ----------
if not stocks_selected and not benches_selected:
    st.info("No tickers selected. Use the sidebar to choose Stocks and/or Benchmarks (left).")
    st.stop()  # nothing else to show

# ---------- Metrics Table ----------
st.subheader("Key Metrics (computed)")
if metrics_df.empty:
    st.write("No metrics available for your selection.")
else:
    st.dataframe(metrics_df.style.format({
        "ROI_%": "{:.2f}",
        "Annual_Return_%": "{:.2f}",
        "Annual_Volatility_%": "{:.2f}",
        "Sharpe": "{:.3f}",
        "Beta_vs_Market": "{:.3f}"
    }), use_container_width=True)

# ---------- Charts ----------
# Stocks price chart (interactive)
if stocks_selected:
    st.subheader("Stock Prices (interactive)")
    price_df = dff_full[stocks_selected].dropna(how='all', axis=1)
    if price_df.empty:
        st.warning("No numeric price data for selected stocks in the chosen date range.")
    else:
        fig = go.Figure()
        for col in price_df.columns:
            fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col], mode='lines', name=col, hovertemplate='%{y:.2f}<br>%{x|%Y-%m-%d}'))
        fig.update_layout(legend_title="Stocks", xaxis_title="Date", yaxis_title="Price", height=450, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No stocks selected - skip stock chart.")

# Benchmarks price chart
if benches_selected:
    st.subheader("Benchmarks (interactive)")
    bench_df = dff_full[benches_selected].dropna(how='all', axis=1)
    if bench_df.empty:
        st.warning("No numeric data for selected benchmarks in the chosen date range.")
    else:
        figb = go.Figure()
        for col in bench_df.columns:
            figb.add_trace(go.Scatter(x=bench_df.index, y=bench_df[col], mode='lines+markers', name=col, hovertemplate='%{y:.2f}<br>%{x|%Y-%m-%d}'))
        figb.update_layout(legend_title="Benchmarks", xaxis_title="Date", yaxis_title="Price", height=350, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(figb, use_container_width=True)
else:
    st.info("No benchmarks selected - skip benchmark chart.")

# Correlation heatmap
if show_corr and len(stocks_selected) > 1:
    st.subheader("Return Correlation (selected stocks)")
    returns = dff_full[stocks_selected].pct_change().dropna()
    if returns.empty:
        st.warning("Not enough returns data to compute correlation (select a longer date range).")
    else:
        corr = returns.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix (daily returns)")
        st.plotly_chart(fig_corr, use_container_width=True)

# Risk-return scatter
if show_risk_return and stocks_selected:
    st.subheader("Risk vs Return (selected stocks)")
    returns = dff_full[stocks_selected].pct_change().dropna()
    if returns.empty:
        st.warning("Not enough returns data to compute risk/return.")
    else:
        ann_vol = returns.std() * np.sqrt(252)
        ann_ret = (1 + returns.mean())**252 - 1
        rr_df = pd.DataFrame({"Ticker": ann_vol.index, "Annual_Volatility_%": ann_vol*100, "Annual_Return_%": ann_ret*100})
        fig_rr = px.scatter(rr_df, x="Annual_Volatility_%", y="Annual_Return_%", text="Ticker", size_max=20,
                            title="Risk vs Return (annualized %)")
        fig_rr.update_traces(textposition='top center')
        st.plotly_chart(fig_rr, use_container_width=True)

# Quick insights
st.markdown("---")
st.subheader("Quick Insights")
ins = []
if not metrics_df.empty:
    # top ROI
    try:
        best = metrics_df['ROI_%'].dropna().idxmax()
        ins.append(f"Top ROI (selected): **{best}** — {metrics_df.loc[best,'ROI_%']:.2f}%")
    except Exception:
        pass
    # highest volatility
    try:
        hi_vol = metrics_df['Annual_Volatility_%'].dropna().idxmax()
        ins.append(f"Highest annual volatility: **{hi_vol}** — {metrics_df.loc[hi_vol,'Annual_Volatility_%']:.2f}%")
    except Exception:
        pass
if not ins:
    st.write("No insights available — try selecting tickers and expanding the date range.")
else:
    for i in ins:
        st.write("-", i)

st.markdown("### Notes")
st.write("""
- All metrics are computed on-the-fly from your uploaded price CSV.  
- Rolling windows and risk-free rate are adjustable in the sidebar.  
- For better Beta results, select an appropriate market benchmark (e.g., DFM or ADX) in Benchmarks.  
""")
