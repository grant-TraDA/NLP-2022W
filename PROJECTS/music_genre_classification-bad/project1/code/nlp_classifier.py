from abc import ABC, abstractmethod
from joblib import dump, load
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn import preprocessing

class NLPClassifier(ABC):
    def __init__(self):
        self.model = None
        self.name = None
    
    def partial_fit(self, X, Y, classes):
        self.model.partial_fit(X, Y, classes)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filename):
        dump(self.model, filename)
    
    def load(self, filename):
        self.model = load(filename)


class NaiveBayes(NLPClassifier):
    def __init__(self):
        self.model = GaussianNB()
        self.name = 'naive-bayes'

class SVM(NLPClassifier):
    def __init__(self):
        self.model = SGDClassifier()
        self.name = 'svm'

class XGBoost(NLPClassifier):    
    def __init__(self):
        self.model = None
        self.name = 'xgboost'
        self.params = {'objective': 'multi:softmax'}
        self.boost_iter = 30
        self.le = preprocessing.LabelEncoder()
    
    def partial_fit(self, X, Y, classes):       
        if self.model is None:
            self.le.fit(classes)
            self.params['num_class'] = len(classes)
            data = xgb.DMatrix(X, label=self.le.transform(Y))
            self.model = xgb.train(self.params, data, self.boost_iter)
        else:
            data = xgb.DMatrix(X, label=self.le.transform(Y))
            self.model = xgb.train(self.params, data, self.boost_iter, xgb_model=self.model)
    
    def predict(self, X):
        return self.le.inverse_transform(self.model.predict(xgb.DMatrix(X)).astype(int))
    
    def save(self, filename):
        self.model.save_model(filename)
    
    def load(self, filename):
        self.model = xgb.Booster()
        self.model.load_model(filename)