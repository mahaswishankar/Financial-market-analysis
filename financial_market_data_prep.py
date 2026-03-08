# ============================================================
# FINANCIAL MARKET ANALYTICS - DATA PREPARATION
# Fetches real stock data and prepares it for Power BI
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, 'data')
charts_dir = os.path.join(script_dir, 'charts')
os.makedirs(data_dir,   exist_ok=True)
os.makedirs(charts_dir, exist_ok=True)

def save(filename):
    path = os.path.join(charts_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved: charts/{filename}")

# ============================================================
# 1. DEFINE STOCKS
# ============================================================
print("=" * 60)
print("STEP 1: Defining Stock Universe")
print("=" * 60)

stocks = {
    # Finance / Banking (JP Morgan relevant!)
    'JPM' : 'JP Morgan Chase',
    'GS'  : 'Goldman Sachs',
    'BAC' : 'Bank of America',
    'MS'  : 'Morgan Stanley',
    # Big Tech
    'AAPL': 'Apple',
    'GOOGL': 'Google',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    # Energy
    'XOM' : 'ExxonMobil',
    'CVX' : 'Chevron',
}

sectors = {
    'JPM' : 'Finance', 'GS'  : 'Finance',
    'BAC' : 'Finance', 'MS'  : 'Finance',
    'AAPL': 'Tech',    'GOOGL': 'Tech',
    'MSFT': 'Tech',    'AMZN': 'Tech',
    'XOM' : 'Energy',  'CVX' : 'Energy',
}

tickers = list(stocks.keys())
START   = '2020-01-01'
END     = datetime.today().strftime('%Y-%m-%d')

print(f"✅ {len(tickers)} stocks across 3 sectors: Finance, Tech, Energy")
print(f"   Period: {START} to {END}")

# ============================================================
# 2. FETCH STOCK DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Fetching Stock Data from Yahoo Finance")
print("=" * 60)

all_data = []

for ticker, name in stocks.items():
    print(f"   📥 Fetching {ticker} ({name})...")
    try:
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        df = df.reset_index()

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

        df['Ticker']  = ticker
        df['Company'] = name
        df['Sector']  = sectors[ticker]
        all_data.append(df)
        print(f"      ✅ {len(df)} rows fetched")
    except Exception as e:
        print(f"      ❌ Failed: {e}")

raw_df = pd.concat(all_data, ignore_index=True)
print(f"\n✅ Total rows fetched: {len(raw_df):,}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

df = raw_df.copy()
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Daily return
df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change() * 100

# Moving averages
df['MA_20']  = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
df['MA_50']  = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50).mean())
df['MA_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200).mean())

# Volatility (20-day rolling std of returns)
df['Volatility_20'] = df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(20).std())

# Price range
df['Daily_Range'] = df['High'] - df['Low']
df['Daily_Range_Pct'] = (df['Daily_Range'] / df['Close']) * 100

# Cumulative return from start
df['Cumulative_Return'] = df.groupby('Ticker')['Close'].transform(
    lambda x: (x / x.iloc[0] - 1) * 100
)

# Trading signal (simple MA crossover)
df['Signal'] = np.where(df['MA_20'] > df['MA_50'], 'BUY', 'SELL')

# Year, Month, Quarter for time analysis
df['Year']    = pd.to_datetime(df['Date']).dt.year
df['Month']   = pd.to_datetime(df['Date']).dt.month
df['Quarter'] = pd.to_datetime(df['Date']).dt.quarter
df['Month_Name'] = pd.to_datetime(df['Date']).dt.strftime('%b')

print("✅ Features created:")
print("   Daily Return, MA 20/50/200, Volatility, Cumulative Return")
print("   Daily Range, Trading Signal, Year/Month/Quarter")

# ============================================================
# 4. QUICK VALIDATION CHARTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Quick Validation Charts")
print("=" * 60)

sns.set_theme(style="darkgrid")

# Chart 1 — Stock price trends
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
finance_stocks = ['JPM', 'GS', 'MS', 'BAC']
for ax, ticker in zip(axes.flatten(), finance_stocks):
    data = df[df['Ticker'] == ticker]
    ax.plot(data['Date'], data['Close'],  color='#3498db', lw=1.5, label='Price')
    ax.plot(data['Date'], data['MA_50'],  color='#e74c3c', lw=1,   label='MA50', linestyle='--')
    ax.plot(data['Date'], data['MA_200'], color='#2ecc71', lw=1,   label='MA200', linestyle='--')
    ax.set_title(f'{stocks[ticker]}', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend(fontsize=8)
plt.suptitle('Finance Sector — Stock Prices with Moving Averages', fontsize=14, fontweight='bold')
plt.tight_layout()
save('01_finance_stocks.png')
plt.show()

# Chart 2 — Cumulative returns comparison
plt.figure(figsize=(14, 7))
colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4']
for (ticker, name), color in zip(stocks.items(), colors):
    data = df[df['Ticker'] == ticker].dropna(subset=['Cumulative_Return'])
    plt.plot(data['Date'], data['Cumulative_Return'], label=ticker, color=color, lw=1.5)
plt.axhline(y=0, color='black', linestyle='--', lw=1)
plt.title('Cumulative Returns Since 2020 — All Stocks', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.legend(ncol=2)
plt.tight_layout()
save('02_cumulative_returns.png')
plt.show()

# Chart 3 — Volatility by sector
latest = df.groupby('Ticker').last().reset_index()
latest['Company'] = latest['Ticker'].map(stocks)
latest['Sector']  = latest['Ticker'].map(sectors)

vol_data = df.groupby(['Ticker', 'Sector'])['Volatility_20'].mean().reset_index()
vol_data['Company'] = vol_data['Ticker'].map(stocks)
vol_data = vol_data.sort_values('Volatility_20', ascending=True)

plt.figure(figsize=(12, 6))
colors_bar = ['#e74c3c' if s == 'Finance' else '#3498db' if s == 'Tech' else '#2ecc71'
              for s in vol_data['Sector']]
plt.barh(vol_data['Company'], vol_data['Volatility_20'], color=colors_bar, edgecolor='black', alpha=0.85)
plt.title('Average 20-Day Volatility by Stock', fontsize=14, fontweight='bold')
plt.xlabel('Volatility (%)')
plt.tight_layout()
save('03_volatility.png')
plt.show()

# ============================================================
# 5. EXPORT CSVs FOR POWER BI
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Exporting CSVs for Power BI")
print("=" * 60)

# Main price data
main_df = df[['Date', 'Ticker', 'Company', 'Sector', 'Open', 'High', 'Low',
              'Close', 'Volume', 'Daily_Return', 'MA_20', 'MA_50', 'MA_200',
              'Volatility_20', 'Cumulative_Return', 'Daily_Range_Pct',
              'Signal', 'Year', 'Month', 'Quarter', 'Month_Name']].copy()

main_df.to_csv(os.path.join(data_dir, '01_stock_prices.csv'), index=False)
print(f"✅ 01_stock_prices.csv — {len(main_df):,} rows (main dataset)")

# Summary by company
summary = df.groupby(['Ticker', 'Company', 'Sector']).agg(
    Current_Price  = ('Close', 'last'),
    Start_Price    = ('Close', 'first'),
    Max_Price      = ('Close', 'max'),
    Min_Price      = ('Close', 'min'),
    Avg_Volume     = ('Volume', 'mean'),
    Avg_Volatility = ('Volatility_20', 'mean'),
    Total_Return   = ('Cumulative_Return', 'last'),
).reset_index().round(2)

summary.to_csv(os.path.join(data_dir, '02_company_summary.csv'), index=False)
print(f"✅ 02_company_summary.csv — company level summary")

# Monthly returns
monthly = df.groupby(['Ticker', 'Company', 'Sector', 'Year', 'Month', 'Month_Name']).agg(
    Monthly_Return  = ('Daily_Return', 'sum'),
    Avg_Close       = ('Close', 'mean'),
    Avg_Volume      = ('Volume', 'mean'),
    Avg_Volatility  = ('Volatility_20', 'mean'),
).reset_index().round(2)

monthly.to_csv(os.path.join(data_dir, '03_monthly_returns.csv'), index=False)
print(f"✅ 03_monthly_returns.csv — monthly aggregated data")

# Sector performance
sector = df.groupby(['Sector', 'Year', 'Quarter']).agg(
    Avg_Return     = ('Daily_Return', 'mean'),
    Avg_Volatility = ('Volatility_20', 'mean'),
    Total_Volume   = ('Volume', 'sum'),
).reset_index().round(2)

sector.to_csv(os.path.join(data_dir, '04_sector_performance.csv'), index=False)
print(f"✅ 04_sector_performance.csv — sector level analysis")

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SUMMARY")
print("=" * 60)
print(f"""
  📦 Data exported to data/ folder:
     01_stock_prices.csv     — {len(main_df):,} rows, main dataset
     02_company_summary.csv  — one row per company
     03_monthly_returns.csv  — monthly aggregated
     04_sector_performance.csv — sector trends

  📊 Charts saved to charts/ folder:
     01_finance_stocks.png
     02_cumulative_returns.png
     03_volatility.png

  🎯 Next Step — Power BI:
     1. Open Power BI Desktop
     2. Click 'Get Data' → 'Text/CSV'
     3. Import all 4 CSV files from the data/ folder
     4. Start building visuals!
""")
print("=" * 60)
print("✅ Data preparation complete! Ready for Power BI.")
print("=" * 60)