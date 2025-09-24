# scripts/2_viz.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_CSV = Path(__file__).parents[1] / "data" / "uae_stock_data.csv"
IM_DIR = Path(__file__).parents[1] / "images"
IM_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_CSV, index_col='Date', parse_dates=True)
    df = df.dropna(how='all', axis=1)
    returns = df.pct_change().dropna()

    # 1) Price trends - plot first 6 tickers (avoid overcrowding)
    top = df.columns[:6]
    fig, ax = plt.subplots(figsize=(12,6))
    df[top].plot(ax=ax)
    ax.set_title("Price Trend (first 6 tickers)")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    fig.savefig(IM_DIR / "price_trends.png", bbox_inches='tight')

    # 2) Correlation heatmap of returns
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="vlag", center=0)
    ax.set_title("Correlation Matrix (Daily Returns)")
    fig.savefig(IM_DIR / "correlation_heatmap.png", bbox_inches='tight')

    # 3) Risk-return scatter
    ann_vol = returns.std() * (252**0.5)
    ann_ret = (1 + returns.mean())**252 - 1
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(ann_vol*100, ann_ret*100)
    for i, ticker in enumerate(returns.columns):
        ax.annotate(ticker, (ann_vol[ticker]*100, ann_ret[ticker]*100))
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Risk vs Return")
    fig.savefig(IM_DIR / "risk_return_scatter.png", bbox_inches='tight')

    print("Saved images to", IM_DIR)

if __name__ == "__main__":
    main()
