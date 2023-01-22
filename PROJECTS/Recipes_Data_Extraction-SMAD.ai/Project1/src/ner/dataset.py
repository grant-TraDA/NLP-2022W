import torch
from itertools import chain


class RecipeNLGIngredientsDataset(torch.utils.data.Dataset):
    def __init__(self, ingredients_list):
        self.ingredients = list(chain.from_iterable(ingredients_list))

    def __len__(self):
        return len(self.ingredients)

    def __getitem__(self, idx):
        return self.ingredients[idx]
