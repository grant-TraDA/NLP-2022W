import pickle

import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

import wandb


CONFIG_FILE = "scripts/dietary_tags_classification/training_config.yaml"
with open(CONFIG_FILE, encoding="utf-8") as f:
    config = yaml.safe_load(f)

if config['use_wandb_logger']:
    wandb.init("dietary-tags-classification-tfidf")

def preprocess_data():
    """Preprocessed data and creates

    :return: returns train and test dataloader and number of tags
    """
    data = pd.read_csv(config["data"]["csv_path"]).dropna()
    if config["data"]["num_examples_to_use_for_training"] > 0:
        data = data.sample(config["data"]["num_examples_to_use_for_training"])
    data.dropna(inplace=True)
    return data


data = preprocess_data()
train_data, test_data = train_test_split(data)
tfidf = TfidfVectorizer(use_idf=True)
train_vectorized = tfidf.fit_transform(train_data.ingredients_list)
test_vectorized = tfidf.transform(test_data.ingredients_list)

lgbm = MultiOutputClassifier(
    LGBMClassifier(class_weight="balanced", n_jobs=-1)
)
lgbm.fit(train_vectorized, train_data[config["data"]["tags_to_use"]])

y_pred = lgbm.predict(test_vectorized)

for i, tag in enumerate(config["data"]["tags_to_use"]):

    output = classification_report(
        y_pred=y_pred[:, i],
        y_true=test_data[tag],
    )
    with open(f"logs/{tag}_results.txt", "w", encoding="utf-8") as f:
        f.write(output)
    if config['use_wandb_logger']:
        wandb.save(f"logs/{tag}_results.txt")

with open('classifier.pkl', 'wb') as f:
    pickle.dump(lgbm, f)
if config['use_wandb_logger']:
    wandb.save('classifier.pkl')
    wandb.finish()
