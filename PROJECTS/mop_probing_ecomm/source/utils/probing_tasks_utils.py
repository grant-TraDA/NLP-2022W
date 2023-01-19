from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")


def test_probing_task(X_train, X_test, y_train, y_test, clf):

    if type(clf) == XGBClassifier:
        clf.fit(X_train, y_train, eval_metric="mlogloss")
    else:
        clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=predictions)
    f_score = f1_score(y_true=y_test, y_pred=predictions, average="macro")

    print(f"Accuracy: {acc}, f_score: {f_score}")

    return (predictions, acc, f_score)
