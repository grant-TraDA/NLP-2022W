import pandas as pd


def words_in_title_check(dataset, key_words):
    train_df_left = dataset[["id_left", "title_left"]]
    train_df_right = dataset[["id_right", "title_right"]]
    train_df_left = train_df_left.drop_duplicates().rename(
        {"id_left": "id", "title_left": "title"}, axis="columns"
    )
    train_df_right = train_df_right.drop_duplicates().rename(
        {"id_right": "id", "title_right": "title"}, axis="columns"
    )
    df_train_all = pd.concat([train_df_right, train_df_left])
    df_train_titles = df_train_all.groupby("id").first().reset_index()

    df_train_titles["keywords_in_title"] = df_train_titles["title"].apply(
        lambda x: any(ele in x.lower() for ele in key_words)
    )

    return df_train_titles
