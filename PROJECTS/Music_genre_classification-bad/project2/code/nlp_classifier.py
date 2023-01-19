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
    def __init__(self):
        self.model = None
        self.name = None

    def partial_fit(self, X, Y, classes):
        self.model.partial_fit(X, Y, classes)

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


class CNNClassifier:
    def partial_fit(self, X, Y, classes):
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


class CNN(CNNClassifier):
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

class BinaryCNN(CNNClassifier):
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

class CNN2Step(CNNClassifier):
    def __init__(self, vec_len, class_count, optimizer, indiv_class):
        self.name = '2_step_cnn'
        self.model1 = BinaryCNN(vec_len, optimizer)
        self.model2 = CNN(vec_len, class_count - 1, optimizer)
        self.indiv_class = indiv_class

    def partial_fit(self, X, Y, classes):
        Y_binary = np.array(Y == self.indiv_class).astype(int)
        self.model1.partial_fit(X.reshape(*X.shape, 1), Y_binary.reshape(-1, 1), classes)
        
        X_other = X[Y != self.indiv_class]
        Y_other = Y[Y != self.indiv_class]
        self.model2.partial_fit(X_other.reshape(*X_other.shape, 1), Y_other.reshape(-1, 1), classes)

    def predict(self, X):
        pred1 = self.model1.predict(X.reshape(*X.shape, 1))
        X2 = X[pred1 == 0]
        pred2 = self.model2.predict(X2.reshape(*X2.shape, 1))
        
        pred = np.zeros(len(X), dtype=int)
        pred[pred1 == 1] = self.indiv_class
        pred[pred1 == 0] = pred2
        
        return pred.flatten()

    def predict_proba(self, X):
        pass
    
    def save(self, filename):
        self.model1.save(f'{filename}1')
        self.model2.save(f'{filename}2')

    def load(self, filename):
        self.model1 = models.load_model(f'{filename}1')
        self.model2 = models.load_model(f'{filename}2')


class Dense(NLPClassifier):
    def __init__(self, hidden_size, class_count, optimizer):
        super().__init__()
        self.name = 'dense'

        model = models.Sequential()
        model.add(layers.Flatten())
        model.add(layers.Dense(hidden_size, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(class_count, activation='softmax'))

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        self.model = model

    def partial_fit(self, X, Y, classes):
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

