# setting path
import os
from os import path

import os, sys

current_dir = os.path.abspath("")
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

    # print(f"Accuracy: {acc}, f_score: {f_score}")

    return (predictions, acc, f_score)


def test_visualize_probing_task(
    FILE_NAME, PROBING_NAME, EMBEDDING_PATH_PROBING, REPO_PATH, TYPE, SIZE, MODEL_TYPE
):
    """Train and test probing task model and save plots to file

    Parameters
    ----------
    FILE_NAME : str
        name of file with embeddings and labels
    PROBING_NAME : str
        name of probing task, this name will be in plot title and plot image name
    EMBEDDING_PATH : str
        
    REPO_PATH : str
        
    TYPE : str
        type of dataset
    SIZE : str
        size of dataset
    MODEL_TYPE : str
        type of model: pre_trained or fine_tuned
    """

    # paths to dataframe with embeddings and labels
    df = path.join(EMBEDDING_PATH_PROBING, FILE_NAME)

    df = pd.read_csv(df)
    X, y = df.drop(["label"], axis=1), df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ----- Multi Class  -----
    clfLR = LogisticRegression(
        multi_class="multinomial", random_state=42, penalty="l1", solver="saga"
    )

    # ----- Binary Class  -----
    clfLR = LogisticRegression(penalty="l1", solver="liblinear")

    clfRF = RandomForestClassifier(random_state=42)

    clfXGB = XGBClassifier(random_state=42)

    pred_rf, acc_rf, f_score_rf = test_probing_task(
        X_train, X_test, y_train, y_test, clfRF
    )

    pred_lr, acc_lr, f_score_lr = test_probing_task(
        X_train, X_test, y_train, y_test, clfLR
    )

    pred_xg, acc_xg, f_score_xg = test_probing_task(
        X_train, X_test, y_train, y_test, clfXGB
    )

    values = [100 * acc_lr, 100 * acc_rf, 100 * acc_xg]
    classifier_name = ["LogisticRegression", "RandomForest", "XGB"]

    fig, ax = plt.subplots()
    bars = ax.bar(classifier_name, values)

    ax.bar_label(bars)
    plt.ylim([0, 100])
    plt.ylabel("accuracy")
    plt.title(
        f"Accuracy Score for probing task:{PROBING_NAME}. Dataset: {TYPE}, {SIZE}, {MODEL_TYPE}"
    )

    PLOT_SAVE_PATH_accuracy = os.path.join(
        REPO_PATH, "project2_output/visualizations", f"{PROBING_NAME}_accuracy.png"
    )
    plt.savefig(PLOT_SAVE_PATH_accuracy, bbox_inches="tight")

    plt.show()

    # -----------------

    values_fscore = [100 * f_score_rf, 100 * f_score_lr, 100 * f_score_xg]
    fig, ax = plt.subplots()
    bars = ax.bar(classifier_name, values_fscore)

    ax.bar_label(bars)
    plt.ylim([0, 100])
    plt.ylabel("F-score")
    plt.title(
        f"F-score for probing task:{PROBING_NAME}. Dataset: {TYPE}, {SIZE}, {MODEL_TYPE}"
    )

    PLOT_SAVE_PATH_fscore = os.path.join(
        REPO_PATH, "project2_output/visualizations", f"{PROBING_NAME}_fscore.png"
    )

    plt.savefig(PLOT_SAVE_PATH_fscore, bbox_inches="tight")

    plt.show()
