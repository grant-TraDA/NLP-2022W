import torch
from src import ner
from pathlib import Path
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import numpy as np
import pickle

SAVE_PATH = Path("dataset/recipe_nlg_ner.pkl")
DEVICE = torch.device("cuda:0")


def main(
    device,
    save_path: Path,
    batch_size: int = 128,
    save_every: int = 10000,
    skip_existing=True,
):
    pipeline = ner.model.load_pipeline(ner.constants.MODEL_NAME, device=device)
    recipe_nlg = pd.read_pickle("dataset/full_dataset.pkl")
    recipe_nlg.ingredients = recipe_nlg.ingredients.apply(literal_eval)

    if skip_existing and save_path.exists():
        with save_path.open("rb") as f:
            all_outputs = pickle.load(f)
        current_example = len(all_outputs)
        recipe_nlg = recipe_nlg.iloc[current_example:].reset_index(drop=True)
    else:
        current_example = 0
        all_outputs = []
    dataset = ner.dataset.RecipeNLGIngredientsDataset(recipe_nlg.ingredients)
    lengths = np.cumsum(recipe_nlg.ingredients.apply(len))
    current_outputs = []
    with torch.inference_mode():
        for i, out in tqdm(
            enumerate(pipeline(dataset, batch_size=batch_size)),
            total=len(dataset),
        ):
            if i >= lengths[current_example]:
                all_outputs.append(current_outputs)
                current_example += 1
                current_outputs = []
                if current_example % save_every == 0:
                    with open(save_path, "wb") as f:
                        pickle.dump(all_outputs, f)
            current_outputs.append(out)
    with open(save_path, "wb") as f:
        pickle.dump(all_outputs, f)


if __name__ == "__main__":
    main(DEVICE, SAVE_PATH, save_every=5000)
