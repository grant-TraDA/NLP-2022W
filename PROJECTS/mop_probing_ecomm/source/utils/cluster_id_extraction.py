import pandas as pd



def extract_ids_and_cluster_ids(train_df):
    """
    extract unique pairs of cluster_id and offer_id
    """
    train_df_left = train_df[["id_left", "title_left"]]
    train_df_right = train_df[["id_right", "title_right"]]
    train_df_left = train_df_left.drop_duplicates().rename({"id_left" : "offer_id", "title_left" : "title"}, axis = 'columns')
    train_df_right = train_df_right.drop_duplicates().rename({"id_right" : "offer_id", "title_right" : "title"}, axis = 'columns')
    df_train_all = pd.concat([train_df_right, train_df_left])

    return df_train_all.drop_duplicates()