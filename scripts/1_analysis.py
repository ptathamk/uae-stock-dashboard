# scripts/1_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path

DATA_CSV = Path(__file__).parents[1] / "data" / "uae_stock_data.csv"
OUT_METRICS = Path(__file__).parents[1] / "data" / "uae_metrics.csv"

RISK_FREE_RATE = 0.031  # example annual risk-free (3.1%). Adjust as needed.

def annualize_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def annualize_return(daily_returns):
    return (1 + daily_returns.mean()) ** 252 - 1

def calculate_sharpe(daily_returns, rf=0.031):
    ann_ret = annualize_return(daily_returns)
    ann_vol = annualize_volatility(daily_returns)
    # Excess return
    if ann_vol == 0:
        return np.nan
    return (ann_ret - rf) / ann_vol

def calculate_beta(returns, market_returns):
    # beta = cov(stock, market) / var(market)
    cov = returns.cov(market_returns)
    var = market_returns.var()
    return cov / var if var != 0 else np.nan

def main():
    df = pd.read_csv(DATA_CSV, index_col='Date', parse_dates=True)
    # drop columns with all NA
    df = df.dropna(how='all', axis=1)
    # daily returns
    returns = df.pct_change().dropna()

    metrics = []
    # identify a market index column if present: common names like DFMGI or ADX (you provided)
    market_cols = [c for c in df.columns if 'DFM' in c.upper() or 'ADX' in c.upper() or c.upper() in ['DFMGI','ADX']]
    market = market_cols[0] if market_cols else None

    for col in df.columns:
        series = df[col].dropna()
        total_roi = (series.iloc[-1] / series.iloc[0] - 1) * 100
        dr = returns[col].dropna() if col in returns.columns else pd.Series(dtype=float)
        ann_vol = float(annualize_volatility(dr) * 100) if not dr.empty else np.nan
        ann_ret = float(annualize_return(dr) * 100) if not dr.empty else np.nan
        sharpe = float(calculate_sharpe(dr, RISK_FREE_RATE)) if not dr.empty else np.nan
        beta = None
        if market and col in returns.columns and market in returns.columns:
            beta = float(calculate_beta(returns[col], returns[market]))
        metrics.append({
            "Ticker": col,
            "ROI_%": round(total_roi, 3),
            "Annual_Return_%": round(ann_ret, 3) if not np.isnan(ann_ret) else np.nan,
            "Annual_Volatility_%": round(ann_vol, 3) if not np.isnan(ann_vol) else np.nan,
            "Sharpe": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
            "Beta_vs_Market": round(beta, 3) if beta is not None else np.nan
        })

    metrics_df = pd.DataFrame(metrics).set_index("Ticker").sort_values(by="ROI_%", ascending=False)
    metrics_df.to_csv(OUT_METRICS)
    print("Saved metrics to", OUT_METRICS)
    print(metrics_df.head(20))

if __name__ == "__main__":
    main()
