import pandas as pd
import json
from ast import literal_eval
from tqdm import tqdm
import numpy as np
DATAPATH = "dataset/recipes_w_search_terms.csv"
# DATAPATH = "dataset/archive/RAW_recipes.csv"
TAGSPATH = "misc/article_tags_mapping.json"
SAVEPATH = "dataset/ingredients_tags.csv"
# SAVEPATH = "dataset/ingredients_article_tags.csv"


def main():
    data = pd.read_csv(DATAPATH)
    data.tags = data.tags.apply(literal_eval)
    with open(TAGSPATH) as f:
        tags_dict = json.load(f)
    tags = {}
    for tag_name, tags_set in tags_dict.items():
        tags_set = set(tags_set)
        tags[tag_name] = data.tags.progress_apply(
            lambda x: len(tags_set.intersection(x)) > 0
        )
    new_dataset = pd.DataFrame.from_dict(tags, orient="columns")
    any_tag_mask = np.any(new_dataset.values, axis=1)
    new_dataset["ingredients"] = data.ingredients_raw_str.progress_apply(
        lambda x: ". ".join(y.strip() for y in literal_eval(x))
    )
    new_dataset = new_dataset[any_tag_mask]
    new_dataset = new_dataset.dropna()
    print(f"Dataset size: {new_dataset.shape}")
    new_dataset.to_csv(SAVEPATH, index=False)


if __name__ == "__main__":
    tqdm.pandas()
    main()
