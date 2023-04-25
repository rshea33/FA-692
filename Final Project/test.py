import yfinance as yf

data = yf.download('AAPL', start='2014-01-02', end='2014-01-09')

print(data)