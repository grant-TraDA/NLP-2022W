import pandas as pd
import numpy as np

np.random.default_rng(123)

INTERESTING_TAGS = [
    "vegetarian",
    "vegan",
    "sweet",
    "seafood",
    "meat",
    "low-sodium",
    "low-carb",
    "low-cholesterol",
    "low-calorie",
    "low-protein",
    "low-saturated-fat",
    "low-fat",
    "dairy",
    "nuts",
    "diabetic",
    "kosher",
]
MAX_LEN = 500


def main():
    """
    Samples from Food.com dataset to have at least 20 examples of each tag
    """
    # loading data
    data = pd.read_csv("dataset/ingredients_tags.csv").dropna()
    # Getting only recipes which are shorter than MAX_LEN
    data = data[data.ingredients.apply(len) < MAX_LEN]
    temp_df = data.copy()
    sampled_df = []
    # Sampling 20 positive examples for each tag.
    for tag in INTERESTING_TAGS:
        sample = temp_df[temp_df[tag]].sample(20)
        temp_df = temp_df.drop(index=sample.index)
        sampled_df.append(sample)
    # Concatenating and shuffling data
    sampled_df = (
        pd.concat(sampled_df, axis=0).sample(frac=1.0).reset_index(drop=True)
    )
    # Saving data
    sampled_df.to_csv("results/llm_sampled_data.csv", index=False)


if __name__ == "__main__":
    main()
