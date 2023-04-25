import pandas as pd 
from EarningsCall import EarningsCall
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    df = pd.DataFrame()

    for year in range(2014, 2024):
        for q in range(1, 5):
            try:
                dataset = f'Raw Data/q{q}_{year}.txt'
                ticker = 'AAPL'
                lags = 5
                earnings_call = EarningsCall(dataset, ticker, lags)
                X = earnings_call.X
                y = earnings_call.y
                X['return'] = y
                df = df.append(X)
                print(df.tail())

            except Exception as e:
                print(f"error: {e}")

    df.to_csv('Clean Data/dataset.csv')


if __name__ == '__main__':
    main()