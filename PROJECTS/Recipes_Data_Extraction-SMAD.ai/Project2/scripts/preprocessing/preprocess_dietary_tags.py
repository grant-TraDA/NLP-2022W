import pandas as pd
import json
from ast import literal_eval
from tqdm import tqdm
import numpy as np

DATAPATH = "dataset/recipes_w_search_terms.csv"
TAGSPATH = "misc/categories_to_tags_mappings.json"
SAVEPATH = "dataset/ingredients_tags.csv"


def main():
    # Loading data
    data = pd.read_csv(DATAPATH)
    # Transforming pandas strings into lists
    data.tags = data.tags.apply(literal_eval)
    # Loading tags mappings
    with open(TAGSPATH) as f:
        tags_dict = json.load(f)
    tags = {}
    # Loop over all dietary tags and doing mapping
    for tag_name, tags_set in tags_dict.items():
        tags_set = set(tags_set)
        # Checking wether dietary tag tags are present int recipe tags
        tags[tag_name] = data.tags.progress_apply(
            lambda x: len(tags_set.intersection(x)) > 0
        )
    # Creating new dataset
    new_dataset = pd.DataFrame.from_dict(tags, orient="columns")
    # Getting only recipes that have at least one positive dietary tag
    any_tag_mask = np.any(new_dataset.values, axis=1)
    # Adding ingredients and ingredients list to dataframe
    new_dataset["ingredients"] = data.ingredients_raw_str.progress_apply(
        lambda x: ". ".join(y.strip() for y in literal_eval(x))
    )
    new_dataset["ingredients_list"] = data.ingredients.progress_apply(
        lambda x: ", ".join(y.strip() for y in literal_eval(x))
    )
    new_dataset = new_dataset[any_tag_mask]
    new_dataset = new_dataset.dropna()
    print(f"Dataset size: {new_dataset.shape}")
    # Saving data
    new_dataset.to_csv(SAVEPATH, index=False)


if __name__ == "__main__":
    tqdm.pandas()
    main()
