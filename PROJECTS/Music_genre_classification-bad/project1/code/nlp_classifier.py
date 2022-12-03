import xgboost as xgb
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from joblib import dump, load
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from tensorflow.keras import layers, models

class NLPClassifier(ABC):
    def __init__(self):
        self.model = None
        self.name = None
    
    def partial_fit(self, X, Y, classes):
        self.model.partial_fit(X, Y, classes)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filename):
        dump(self.model, f'{filename}.joblib')
    
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
    def __init__(self, class_count, boost_iter=30):
        self.model = None
        self.name = 'xgboost'
        self.params = {'objective': 'multi:softmax', 'num_class': class_count}
        self.boost_iter = boost_iter
    
    def partial_fit(self, X, Y, classes):
        data = xgb.DMatrix(X, label=Y)
        self.model = xgb.train(self.params, data, self.boost_iter, xgb_model=self.model)
    
    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X)).astype(int)
    
    def save(self, filename):
        self.model.save_model(f'{filename}.json')
    
    def load(self, filename):
        self.model = xgb.Booster()
        self.model.load_model(filename)

class CNN(NLPClassifier):
    def __init__(self, vec_len, class_count, optimizer):
        self.name = 'cnn'

        model = models.Sequential()
        model.add(layers.Conv1D(8, 3, padding='same', activation='relu', input_shape=(vec_len, 1)))
        model.add(layers.MaxPooling1D(2, strides=2))
        model.add(layers.Conv1D(16, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(2, strides=2))
        model.add(layers.Conv1D(32, 3, padding='same', activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(class_count))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model = model

    def partial_fit(self, X, Y, classes):
        X_proc = X.reshape(*X.shape, 1)
        Y_proc = Y.reshape(-1, 1)
        self.model.fit(X_proc, Y_proc)

    def predict(self, X):
        pred = np.argmax(self.model.predict(X.reshape(*X.shape, 1)), axis=1)
        return pred.flatten()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = models.load_model(filename)
