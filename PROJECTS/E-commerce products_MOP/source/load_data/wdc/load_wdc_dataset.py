import pandas as pd
import os
from sentence_transformers.readers import InputExample
from pathlib import Path
import requests
from zipfile import ZipFile

import ssl


class EnglishDatasetLoader:
    MAIN_DIR_PATH = (
        "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2"
    )

    @staticmethod
    def load_train(type: object, size: object) -> pd.DataFrame:
        """Loads the training dataset from WDC website
        Args:
            type (object): dataset type: computers, cameras, watches, shoes, all
            size (object): dataset size: small, medium, large, xlarge
        Returns:
            pd.DataFrame: training dataset
        """
        p = Path(os.path.join("trainsets", f"{type}_train"))
        p.mkdir(parents=True, exist_ok=True)
        dataset_path = f"{p}/{type}_train_{size}.json.gz"
        if not os.path.exists(dataset_path):
            zip_path = f"{p}.zip"
            url = f"{EnglishDatasetLoader.MAIN_DIR_PATH}/trainsets/{type}_train.zip"
            r = requests.get(url, allow_redirects=True)
            open(zip_path, "wb").write(r.content)
            with ZipFile(zip_path, "r") as zip:
                zip.extractall(path=p)

        df = pd.read_json(dataset_path, compression="gzip", lines=True)
        return df.reset_index()

    @staticmethod
    def load_test(type: object) -> pd.DataFrame:
        """Loads the test dataset form repository
        Args:
            type (object): dataset type: computers, cameras, watches, shoes, all
        Returns:
            pd.DataFrame: test dataset
        """
        ssl._create_default_https_context = ssl._create_unverified_context

        path = f"{EnglishDatasetLoader.MAIN_DIR_PATH}/goldstandards/{type}_gs.json.gz"
        df = pd.read_json(path, compression="gzip", lines=True)
        return df.reset_index()


def get_samples(df, features_to_concat=("title", "description")):
    samples = []
    for index, row in df.iterrows():
        label = (row["cluster_id_right"] == row["cluster_id_left"]) * 1.0
        sentence1 = ""
        sentence2 = ""
        guid = f"{row['id_right']}_{row['id_left']}"
        for f in features_to_concat:
            f_val_right = row[f"{f}_right"]
            f_val_left = row[f"{f}_left"]
            sentence1 += f_val_right + " " if f_val_right is not None else ""
            sentence2 += f_val_left + " " if f_val_left is not None else ""

        inp_example = InputExample(guid=guid, texts=[sentence1, sentence2], label=label)
        samples.append(inp_example)
    return samples


def get_wdc_dataset(
    dataset_type="cameras",
    dataset_size="small",
    is_train=True,
    features_to_concat=("title", "description"),
):
    if is_train:
        df = EnglishDatasetLoader.load_train(dataset_type, dataset_size)
    else:
        df = EnglishDatasetLoader.load_test(dataset_type)
    samples = get_samples(df, features_to_concat)
    return samples

