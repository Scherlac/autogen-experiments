# filename: stock_returns_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Define the stock symbols and the date range
symbols = ['NVDA', 'TSLA']
start_date = '2024-01-01'

# Fetch the stock data
data = yf.download(symbols, start=start_date)

# Check the structure of the downloaded data
print(data.head())  # This will help us understand the structure of the DataFrame

# Select 'Close' prices for NVDA and TSLA
data_close = data['Close']

# Calculate daily returns with fill_method set to None
returns = data_close.pct_change(fill_method=None).dropna()

# Calculate cumulative returns
cumulative_returns = (1 + returns).cumprod() - 1

# Plotting the cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns.index, cumulative_returns['NVDA'], label='NVDA')
plt.plot(cumulative_returns.index, cumulative_returns['TSLA'], label='TSLA')

plt.title('Cumulative Returns of NVDA vs TSLA YTD (2024)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid()
plt.savefig('stock_returns_ytd.png')
