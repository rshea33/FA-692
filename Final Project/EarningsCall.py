import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class EarningsCall:

    """This class is used to store the data from each earnings call"""

    def __init__(self, dataset, ticker, lags) -> None:

        self.dataset = dataset
        self.ticker = ticker
        self.lags = lags

        self.date_str_ = pd.read_csv(self.dataset, nrows=1, header=None).values[0][0]
        self.date = datetime.strptime(self.date_str_, '%Y-%m-%d')

        self.start_date = self.date - pd.DateOffset(days=3*self.lags)  # multiply by 3 to avoid NaNs
        self.start_date_str_ = self.start_date.strftime('%Y-%m-%d')
        self.future_date_ = self.date + pd.DateOffset(days=14)
        self.future_date_str_ = self.future_date_.strftime('%Y-%m-%d')

        self.fin_data = yf.download(self.ticker, start=self.start_date_str_, end=self.future_date_str_)

        self.log_returns = self.get_log_returns()
        self.delta_volume = self.get_delta_volume()

        
        for i in range(0, self.lags+1):  # lag returns, change in volume
            self.fin_data[f'log_returns_{i}'] = self.log_returns.shift(i)
            self.fin_data[f'delta_volume_{i}'] = self.delta_volume.shift(i)
        self.fin_data = self.fin_data[self.date_str_:]

        self.X = self.fin_data.drop([
            'Open',
            'High',
            'Low',
            'Close',
            'Adj Close',
            'Volume',
            'log_returns_0',
            'delta_volume_0'
            ], axis=1).loc[self.date_str_]
        
        self.y = self.get_cumulative_returns()

        # Add body of earnings call
        with open(self.dataset, 'r') as f:
            next(f)  # first line, this is self.date_str_
            self.body = [line for line in f if line.strip()]
            self.body = [line.replace('\n', '') for line in self.body]

        # Add sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        self.sentiment = analyzer.polarity_scores(' '.join(self.body))

        for key, value in self.sentiment.items():
            self.X[key] = value
        self.X['pos-neg'] = self.X['pos'] - self.X['neg']  # add new feature

    def get_log_returns(self) -> pd.Series:
        return np.log(self.fin_data['Adj Close'] /
                      self.fin_data['Adj Close'].shift(1))
    
    def get_delta_volume(self) -> pd.Series:
        return self.fin_data['Volume'].pct_change()
    
    def get_hml(self) -> pd.Series:
        hml = self.fin_data['High'] - self.fin_data['Low']
        return (hml / self.fin_data['Low'])[1:]  # slight normalization w.r.t. stock price  
    
    def get_cumulative_returns(self) -> float:
        return (self.fin_data['Adj Close'][-1] - self.fin_data['Adj Close'][0]) / \
                self.fin_data['Adj Close'][0]


def main():
    dataset = 'Raw Data/q1_2014.txt'
    call = EarningsCall(dataset, 'AAPL', 5)
    print(call.date)
    # print(call.day_before_str_)
    print(call.future_date_str_)
    # print(call.body)
    print(call.fin_data)
    # print(call.get_log_returns())
    # print(call.get_delta_volume())
    # print(call.get_hml())
    # print(call.get_cumulative_returns())
    print(call.X)
    print(call.y)


if __name__ == '__main__':
    main()
