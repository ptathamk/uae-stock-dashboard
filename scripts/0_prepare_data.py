# scripts/0_prepare_data.py
import pandas as pd
import pathlib
from datetime import datetime

DATA_DIR = pathlib.Path(__file__).parents[1] / "data"
OUT_CSV = DATA_DIR / "uae_stock_data.csv"

def read_excel_price_file(path):
    """
    Attempts to read an Excel workbook or CSV and return a DataFrame with Date index
    and a single price series column named by the ticker (inferred from filename).
    Accepts either:
      - an Excel workbook with a sheet per ticker, or
      - one file per ticker with columns Date and Adj Close or Close.
    """
    name = path.stem.upper().replace(" ", "_")
    try:
        # Try reading CSV
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, parse_dates=['Date'], dayfirst=False)
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close' if 'Close' in df.columns else df.columns[-1]
            df = df[['Date', col]].rename(columns={col: name}).set_index('Date').sort_index()
            return df
        # Excel
        xls = pd.ExcelFile(path)
        # Heuristic: if excel has multiple sheets and one matches ticker, try each sheet
        if len(xls.sheet_names) == 1:
            df = pd.read_excel(xls, sheet_name=0, parse_dates=['Date'])
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close' if 'Close' in df.columns else df.columns[-1]
            df = df[['Date', col]].rename(columns={col: name}).set_index('Date').sort_index()
            return df
        else:
            # If multi-sheet workbook, try each sheet and guess ticker from sheet name
            frames = []
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, parse_dates=['Date'])
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close' if 'Close' in df.columns else None
                if col:
                    ticker = sheet.upper().replace(" ", "_")
                    df = df[['Date', col]].rename(columns={col: ticker}).set_index('Date').sort_index()
                    frames.append(df)
            if len(frames) == 1:
                return frames[0]
            elif frames:
                return pd.concat(frames, axis=1)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
    return None

def main():
    files = list(DATA_DIR.glob("*.*"))
    if not files:
        print("No files found in data/ — put your Excel/CSV files there, each file per ticker or a multi-sheet workbook.")
        return

    all_frames = []
    for f in files:
        df = read_excel_price_file(f)
        if df is None:
            print(f"Skipping {f.name}")
        else:
            all_frames.append(df)

    if not all_frames:
        print("No valid dataframes loaded. Check file format.")
        return

    # Merge on Date index (outer join to preserve all dates)
    merged = pd.concat(all_frames, axis=1, join='outer').sort_index()
    # Forward fill missing (use ffill to carry prices across holidays) then drop any remaining NA
    merged = merged.fillna(method='ffill').dropna(how='all')
    # Save to CSV
    merged.to_csv(OUT_CSV, index=True, date_format="%Y-%m-%d")
    print(f"Saved consolidated CSV to {OUT_CSV}. Columns (tickers): {list(merged.columns)}")
    print(merged.head())

if __name__ == "__main__":
    main()
