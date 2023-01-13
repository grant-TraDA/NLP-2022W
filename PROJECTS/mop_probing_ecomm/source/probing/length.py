import pandas as pd
from source.emb_extr_res.emb_extr_res import get_embeddings_df



def prepare_probing_len(df, train_embeddings_path):

    df["nr_of_chars"] = df["title"].apply(lambda x : len(x))
    df["nr_of_words"] = df["title"].apply(lambda x : len(x.split()))

    nr_of_words_bins = [0, 10, 15, 20, 100]
    nr_of_chars_bins = [0, 50, 75, 100, 500]
    df['nr_of_chars_bins'] = pd.cut(x=df['nr_of_chars'], bins=nr_of_chars_bins, labels=[0, 1, 2, 3])

    df['nr_of_words_bins'] = pd.cut(x=df['nr_of_words'], bins=nr_of_words_bins, labels=[0, 1, 2, 3])

    

    embedding_train_df_all = get_embeddings_df(train_embeddings_path)

    probing_df_chars = pd.merge(df[["id", "nr_of_chars_bins" ]], embedding_train_df_all, left_on = "id", right_on = 'id')
    probing_df_chars = probing_df_chars.rename({"nr_of_chars_bins" : "label"}, axis=1)

    probing_df_words = pd.merge(df[["id", "nr_of_words_bins" ]], embedding_train_df_all, left_on = "id", right_on = 'id')
    probing_df_words = probing_df_words.rename({"nr_of_words_bins" : "label"}, axis=1)

    return probing_df_chars, probing_df_words
