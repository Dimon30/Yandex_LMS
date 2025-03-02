from numpy import median
import pandas as pd
from sklearn.base import ClassifierMixin

class RubricCityMedianClassifier(ClassifierMixin):
    # Predicts the rounded (just in case) median of y_train
    def __init__(self, *, param=1):
        self.dict_pro = None
        self.is_fitted = None
        self.result = None
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
        data = pd.DataFrame({'modified_rubrics': X['modified_rubrics'],
                             'city': X['city'],
                             'average_bill': y})
        city = 0
        by_city = 1
        msk = 0
        spb = 1
        data_by_city = pd.DataFrame(data.groupby('city'))[by_city]
        medians_msk = data_by_city[msk].groupby("modified_rubrics")["average_bill"].median().to_dict()
        medians_spb = data_by_city[spb].groupby("modified_rubrics")["average_bill"].median().to_dict()
        self.dict_pro = {"msk": medians_msk, "spb": medians_spb}
        self.is_fitted = True
        return self

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        self.result = [self.dict_pro[city][rubric] for city, rubric in zip(X["city"], X["modified_rubrics"])]
        return self.result