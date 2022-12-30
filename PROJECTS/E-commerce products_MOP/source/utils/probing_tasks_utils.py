from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")
import numpy as np


def test_probing_task(train, test, clf):

    X_train, y_train = (train.drop(["label"], axis=1), np.array(train["label"]))
    X_test, y_test = (test.drop(["label"], axis=1), np.array(test["label"]))

    if type(clf) == XGBClassifier:
        clf.fit(X_train, y_train, eval_metric="mlogloss")
    else:
        clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=predictions)
    f_score = f1_score(y_true=y_test, y_pred=predictions, average="macro")

    print(f"Accuracy: {acc}, f_score: {f_score}")

    return (predictions, acc, f_score)
