from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings
from rapidfuzz.distance import Levenshtein
import jaro

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


def compute_string_dist_for_pairs(df, col_prefix="title", new_col_name=None):

    jaro_metric = df.apply(
        lambda x: jaro.jaro_winkler_metric(
            x[f"{col_prefix}_right"], x[f"{col_prefix}_left"]
        ),
        axis=1,
    )
    lev_metric = df.apply(
        lambda x: Levenshtein.normalized_similarity(
            x[f"{col_prefix}_right"], x[f"{col_prefix}_left"]
        ),
        axis=1,
    )
    jacard_metric = df.apply(
        lambda x: jaccard_similarity(x[f"{col_prefix}_right"], x[f"{col_prefix}_left"]),
        axis=1,
    )

    return jaro_metric, lev_metric, jacard_metric


def jaccard_similarity(x, y):
    sentences = [x, y]
    sentences = [sent.lower().split(" ") for sent in sentences]

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)

