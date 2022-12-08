from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import numpy as np


def test_probing_task_xgb(train, test):

    """
    Returns: returns predictions, accuracy
    """
    le = preprocessing.LabelEncoder()
    train_labels = le.fit_transform(train.iloc[:, -1])
    test_labels = le.transform(test.iloc[:, -1])

    model_1 = XGBClassifier()
    model_1.fit(train.iloc[:, :-1], train_labels)
    pred_1 = model_1.predict(test.iloc[:, :-1])

    return pred_1  # , accuracy_score(y_true=test_labels, y_pred=predictions)


def test_probing_task_random_forest(train, test):

    """
    Returns: returns predictions, accuracy
    """
    X_train, y_train = train.drop(["label"], axis=1), np.array(train["label"])
    X_test, y_test = test.drop(["label"], axis=1), np.array(test["label"])

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    return (
        predictions,
        accuracy_score(y_true=y_test, y_pred=predictions),
        f1_score(y_true=y_test, y_pred=predictions, average="macro"),
    )


def test_probing_task_logistic_regression(train, test):
    X_train, y_train = train.drop(["label"], axis=1), np.array(train["label"])
    X_test, y_test = test.drop(["label"], axis=1), np.array(test["label"])

    clf = LogisticRegression(
        multi_class="multinomial", random_state=42, penalty="l1", solver="saga"
    )

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    return (
        predictions,
        accuracy_score(y_true=y_test, y_pred=predictions),
        f1_score(y_true=y_test, y_pred=predictions, average="macro"),
    )
