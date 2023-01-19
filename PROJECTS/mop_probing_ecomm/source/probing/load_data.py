import os, sys

# current_dir = os.path.abspath("")
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from source.load_data.wdc.load_wdc_dataset import EnglishDatasetLoader
from source.emb_extr_res.emb_extr_res import get_embeddings_df
from source.load_data.natural.natural import get_natural_dataset
import pandas as pd

def load_files_probing_tasks(EMBEDDING_PATH, DATA_PATH, TYPE, SIZE):

    # paths to embeddings
    test_embeddings_path = os.path.join(EMBEDDING_PATH, r"test_embeddings.csv")
    train_embeddings_path = os.path.join(EMBEDDING_PATH, r"train_embeddings.csv")

    # load embeddings

    embedding_train_df = get_embeddings_df(train_embeddings_path)
    embedding_train_df

    # load original dataset
    if TYPE == "natural":
        try:
            train_df = pd.read_csv(os.path.join(DATA_PATH, "df_train.csv"))
        except:

            train_df, _ = get_natural_dataset(DATA_PATH, "train_all.csv")
    elif TYPE == "computers":
        train_df = EnglishDatasetLoader.load_train(TYPE, SIZE)
    elif TYPE == "cameras":
        train_df = EnglishDatasetLoader.load_train(TYPE, SIZE)

    return test_embeddings_path, train_embeddings_path, embedding_train_df, train_df
