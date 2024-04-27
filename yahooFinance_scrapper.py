import yfinance as yf
import pandas as pd
stock_symbol = '^DJI'
start_date = '2012-01-01'
end_date = '2024-01-01'
df = yf.download(stock_symbol, start=start_date, end=end_date)
print(df.head())
df=pd.DataFrame(df)
df.to_csv('DJI.csv')