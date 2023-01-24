from sklearn.model_selection import train_test_split
import pandas as pd
import os


def get_natural_dataset(path, input_file_name):

    df_natural = pd.read_csv(os.path.join(path, input_file_name))
    df_natural = df_natural.sample(frac=0.027, random_state=42)
    y_natural = df_natural["is_duplicate"]
    X_natural = df_natural.drop("is_duplicate", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_natural, y_natural, random_state=42, test_size=0.2
    )
    X_train["is_duplicate"] = y_train.values
    X_test["is_duplicate"] = y_test.values

    X_train = X_train.rename(
        {
            "question1": "title_left",
            "question2": "title_right",
            "qid1": "id_left",
            "qid2": "id_right",
            "is_duplicate": "label",
        },
        axis=1,
    )

    X_test = X_test.rename(
        {
            "question1": "title_left",
            "question2": "title_right",
            "qid1": "id_left",
            "qid2": "id_right",
            "is_duplicate": "label",
        },
        axis=1,
    )

    X_train.to_csv(os.path.join(path, "df_train.csv"), index=False)
    X_test.to_csv(os.path.join(path, "df_test.csv"), index=False)

    return X_train, X_test
