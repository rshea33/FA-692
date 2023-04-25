import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

from sklearn import metrics
from sklearn.linear_model import LinearRegression

class Model:

    def __init__(self, model_type):
        if model_type not in ['lr', 'rf']:
            raise ValueError('Model type must be either "lr" or "rf"')
        self.model_type = model_type
        self.model = None

        self.dataset = pd.read_csv('Clean Data/dataset.csv').drop('Unnamed: 0', axis=1)
        self.X = self.dataset.drop(columns=['return'])
        self.y = self.dataset['return']

    def train(self):
        if self.model is not None:
            raise ValueError('Model already trained')
        if self.model_type == 'lr':
            self.model = LinearRegression()
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor()

        self.model.fit(self.X, self.y)

    def predict(self, X):
        if self.model is None:
            raise ValueError('Model not trained')
        return self.model.predict(X)
    
    def evaluate(self):
        if self.model is None:
            raise ValueError('Model not trained')
        loo = LeaveOneOut()
        loo.get_n_splits(self.X)
        y_true = []
        y_pred = []
        for train_index, test_index in loo.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            y_true.append(y_test.values[0])
            y_pred.append(self.model.predict(X_test)[0])
        return metrics.mean_squared_error(y_true, y_pred)


if __name__ == '__main__':
    model = Model('rf')
    # print(model.dataset)
    model.train()
    print(model.evaluate())

    model2 = Model('lr')
    # model2.train()
    print(model.evaluate())




