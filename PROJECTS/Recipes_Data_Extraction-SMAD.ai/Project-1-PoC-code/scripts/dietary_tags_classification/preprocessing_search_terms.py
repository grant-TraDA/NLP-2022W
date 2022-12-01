import pandas as pd
import json
from ast import literal_eval
from tqdm import tqdm

DATAPATH = "dataset/recipes_w_search_terms.csv"
TAGSPATH = "misc/selected_tags.json"
SAVEPATH = "dataset/ingredients_search_tags.csv"


def main():
    data = pd.read_csv(DATAPATH)
    data.search_terms = data.search_terms.apply(literal_eval)
    with open(TAGSPATH) as f:
        tags_dict = json.load(f)
    tags = {}
    for tag_name, tags_set in tags_dict.items():
        tags_set = set(tags_set)
        tags[tag_name] = data.search_terms.progress_apply(
            lambda x: len(tags_set.intersection(x)) > 0
        )
    new_dataset = pd.DataFrame.from_dict(tags, orient="columns")
    new_dataset["ingredients"] = data.ingredients_raw_str.progress_apply(
        lambda x: ". ".join(y.strip() for y in literal_eval(x))
    )
    new_dataset = new_dataset.dropna()
    new_dataset.to_csv(SAVEPATH, index=False)


if __name__ == "__main__":
    tqdm.pandas()
    main()
