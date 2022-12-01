import torch
import numpy as np

class DietaryTagsDataset(torch.utils.data.Dataset):
    """Dataset class for dietary tags"""

    def __init__(self, texts, tags) -> None:
        super().__init__()
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        return self.texts[idx], self.tags[idx]


def get_collate_function(tokenizer):
    """Creates collate function for dataloader with help of tokenizer

    :param tokenizer: tokenizer which is used for model
    """

    def collate_function(batch):
        texts, tags = zip(*batch)
        texts = tokenizer(texts, return_tensors="pt", padding=True)
        tags = torch.tensor(np.array(tags))
        return texts, tags

    return collate_function
