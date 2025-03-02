import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin

class CityMeanRegressor(RegressorMixin):
    def __init__(self, *, param=1):
        self.result = None
        self.city_mean = None
        self.is_fitted = None
        self.param = param
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        if X is None:
            return self
        if y is None:
            return self
        data = pd.DataFrame({'city': X['city'], 'average_bill': y})
        city_avg = data.groupby('city')['average_bill'].mean()
        self.city_mean = city_avg
        self.result = None
        self.is_fitted = True
        return self

    def city_average_predictor(self, city):
        return self.city_mean.get(city, np.nan)
    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        if X is None:
            return self.result
        self.result = [self.city_mean.get(city, np.nan) for city in X["city"]]
        return self.result