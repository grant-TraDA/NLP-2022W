import click
import logging
import datetime
import math

from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

# source code imports
import os, sys
current_dir = os.path.abspath('')
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import SIMILARITY_PATH, EMBEDDING_PATH, PRETRAIN_OUTPUT_PATH
from source.emb_extr_res.emb_extr_res import get_embeddings_df, get_pairs_similarity_df, get_pretrain_agg_similarity
from source.load_data.wdc.load_wdc_dataset import EnglishDatasetLoader
from source.probing.brand_names import prepare_brands_list, brands_in_title_check, prepare_new_dataset, drop_brands
from source.probing.length import prepare_probing_len
from source.probing.words import words_in_title_check




@click.command()

# Required.
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--inputdir', help='Directory where computed embeddings are located', metavar='DIR', required=False)
@click.option('--dataset_type', help='WDC dataset category', metavar='STR', required=True, default='cameras', show_default=True)
@click.option('--dataset_size', help='WDC dataset size', metavar='STR', required=True, default='medium', show_default=True)

def main(outdir, inputdir, dataset_type, dataset_size):
    # paths to results
    test_embeddings_path = path.join(inputdir, r'test_embeddings.csv')
    train_embeddings_path = path.join(inputdir, r'train_embeddings.csv')

    train_df = EnglishDatasetLoader.load_train(dataset_type, dataset_size)

    brands_to_drop = [',', 'd','memory',  'photo', 'co', 'usa',  'power',  'digital', 'camera', 'cam',  'hd',  'a',  'inc',  'le',  'film',  'case',  'pro', 'cameras']
    brands = prepare_brands_list(train_df, brands_to_drop)

    train_df_left = train_df[["id_left", "title_left"]]
    train_df_right = train_df[["id_right", "title_right"]]
    train_df_left = train_df_left.drop_duplicates().rename({"id_left" : "id", "title_left" : "title"}, axis = 'columns')
    train_df_right = train_df_right.drop_duplicates().rename({"id_right" : "id", "title_right" : "title"}, axis = 'columns')
    df_train_all = pd.concat([train_df_right, train_df_left])
    df_train_titles = df_train_all.groupby("id").first().reset_index()

    probing_df_chars, probing_df_words = prepare_probing_len(df_train_titles, train_embeddings_path)

    new_dataset, ids_removed_brands = prepare_new_dataset(train_df, brands)

    probing_task_df = new_dataset[new_dataset["changed"] == True].drop("changed", axis=1)

    probing_df_words.to_csv(path.join(outdir, r'probing_tasks\dataset\probing_df_words.csv'))
    probing_df_chars.to_csv(path.join(outdir, r'probing_tasks\dataset\probing_df_chars.csv'))

    key_words = ['camera', 'digital', 'len']
    df_words = words_in_title_check(train_df, key_words)

    embedding_train_df_all = get_embeddings_df(train_embeddings_path)

    probing_df_key_words = pd.merge(df_words[["id", "brand_in_title" ]], embedding_train_df_all, left_on = "id", right_on = 'id')
    probing_df_key_words = probing_df_key_words.rename({"brand_in_title" : "label"}, axis=1)
    probing_df_key_words = probing_df_key_words.drop("id", axis=1)

    probing_df_key_words.to_csv(path.join(outdir, r'probing_tasks\dataset\probing_df_key_words.csv', index=False))

    brands_in_title_df = brands_in_title_check(new_dataset, brands)
    deleted_ids = brands_in_title_df[brands_in_title_df["brand_in_title"]==True]["id"].values

    embedding_train_df = get_embeddings_df(path.join(EMBEDDING_PATH, r'train_embeddings_removed_brands1.csv'))
    embedding_train_df_all = get_embeddings_df(train_embeddings_path)

    new_emb = embedding_train_df_all[~embedding_train_df_all["id"].isin(ids_removed_brands)] 
    new_emb1 = embedding_train_df[embedding_train_df["id"].isin(ids_removed_brands)]
    new = pd.concat([new_emb1, new_emb])
    new["label"] = new["id"].isin(deleted_ids)

    new.to_csv(path.join(outdir, r'probing_tasks\dataset\train_brand_names.csv', index=False))


if __name__ == "__main__":
    main()
