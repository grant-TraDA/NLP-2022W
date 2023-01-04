import xgboost as xgb
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from joblib import dump, load
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from tensorflow.keras import layers, models
tf.get_logger().setLevel('ERROR')

class NLPClassifier(ABC):
    def __init__(self, classes):
        self.model = None
        self.name = None
        self.classes = classes

    def partial_fit(self, X, Y):
        self.model.partial_fit(X, Y, self.classes)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

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

    def partial_fit(self, X, Y):
        data = xgb.DMatrix(X, label=Y)
        self.model = xgb.train(self.params, data, self.boost_iter, xgb_model=self.model)

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X)).astype(int)

    def save(self, filename):
        self.model.save_model(f'{filename}.json')

    def load(self, filename):
        self.model = xgb.Booster()
        self.model.load_model(filename)

class CNNClasifier(NLPClassifier):

    def partial_fit(self, X, Y):
        X_proc = X.reshape(*X.shape, 1)
        Y_proc = Y.reshape(-1, 1)
        self.model.fit(X_proc, Y_proc)

    def predict(self, X):
        pred = np.argmax(self.model.predict(X.reshape(*X.shape, 1)), axis=1)
        return pred.flatten()

    def predict_proba(self, X):
        pred = self.model.predict(X.reshape(*X.shape, 1))
        return pred.flatten()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = models.load_model(filename)

class CNN(CNNClasifier):
    def __init__(self, vec_len, class_count, optimizer):
        self.name = 'cnn'

        model = models.Sequential()
        model.add(layers.Conv1D(8, 3, padding='same', activation='relu', input_shape=(vec_len, 1)))
        model.add(layers.MaxPooling1D(2, strides=2))
        model.add(layers.Dropout(0.05))
        model.add(layers.Conv1D(16, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(2, strides=2))
        model.add(layers.Dropout(0.05))
        model.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(class_count, activation='softmax'))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model = model

class BinaryCNN(CNNClasifier):
    def __init__(self, vec_len, optimizer):
        self.name = 'binary_cnn'

        model = models.Sequential()
        model.add(layers.Conv1D(8, 3, padding='same', activation='relu', input_shape=(vec_len, 1)))
        model.add(layers.MaxPooling1D(2, strides=2))
        model.add(layers.Dropout(0.05))
        model.add(layers.Conv1D(16, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=['accuracy'])

        self.model = model
        
    def predict(self, X):
        pred = self.model.predict(X.reshape(*X.shape, 1))
        return np.where(pred > 0.5, 1, 0).flatten()

    def predict_proba(self, X):
        pass
