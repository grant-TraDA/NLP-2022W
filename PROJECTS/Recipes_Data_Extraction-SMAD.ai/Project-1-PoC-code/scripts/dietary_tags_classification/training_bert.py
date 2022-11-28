import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from src.dietary_tags_classification.data import (
    DietaryTagsDataset,
    get_collate_function,
)
from pytorch_lightning.loggers import WandbLogger
from src.dietary_tags_classification.model import DietaryTagsClassifierTrainer

CONFIG_FILE = "scripts/dietary_tags_classification/training_config.yaml"
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)


torch.manual_seed(123)
np.random.seed(123)


def preprocess_data(tokenizer):
    """Preprocessed data and creates dataloaders for training

    :param tokenizer: tokenizer to be used in collate function
    :return: returns train and test dataloader and number of tags
    """
    data = pd.read_csv(config["data"]["csv_path"]).dropna()
    if config["data"]["num_examples_to_use_for_training"] > 0:
        data = data.sample(config["data"]["num_examples_to_use_for_training"])
    texts = data.iloc[:, config["data"]["ingredients_index"]].values
    tags = data[config["data"]["tags_to_use"]].astype(float)
    tags_names = list(tags.columns)
    tags = tags.values
    train_texts, test_texts, train_tags, test_tags = train_test_split(
        texts, tags, test_size=config["data"]["test_size"]
    )
    train_dataset = DietaryTagsDataset(train_texts, train_tags)
    test_dataset = DietaryTagsDataset(test_texts, test_tags)
    collate_fn = get_collate_function(tokenizer)
    train_dataloder = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        **config["dataloader_params"]
    )
    test_dataloder = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        **config["dataloader_params"]
    )
    return train_dataloder, test_dataloder, tags_names


def main():
    tokenizer = BertTokenizer.from_pretrained(
        config["tokenizer"]["pretrained_path"]
    )
    train_dataloader, test_dataloader, tags_names = preprocess_data(tokenizer)
    model = DietaryTagsClassifierTrainer(
        tags_names=tags_names, **config["model"]
    )
    wandb_logger = WandbLogger(
        project="dietary_tags_classification", config=config
    )
    wandb_logger.watch(model)
    trainer = Trainer(logger=wandb_logger, **config["trainer_params"])
    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.save_checkpoint("models/current_model")


if __name__ == "__main__":
    main()
