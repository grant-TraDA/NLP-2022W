import requests
import ast
import time
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import pandas as pd

from utils import get_nlg_ingredient_dictionary


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--input_path', '-i', type=Path, help='Path to recipeNLG dataset .csv')
    parser.add_argument('--output_path', '-o', type=Path, help='Path to output .csv with nutritional information')
    parser.add_argument('--api_key', type=str, help='FOODDATA CENTRAL API KEY')

    return parser.parse_args()


def main(args):
    Path(args.output_path.parent).mkdir(exist_ok=True, parents=True)
    ret_df_columns = ["ingredient", "description", "fat", "fat_unit", "protein", "protein_unit",
                      "carbohydrates", "carbohydrates_unit", "energy", "energy_unit"]

    if args.output_path.exists():
        ret_df = pd.read_csv(args.output_path)
    else:
        ret_df = pd.DataFrame([], columns=ret_df_columns)

    nutrient_ids = {1003: "protein", 1004: "fat", 1005: "carbohydrates", 1008: "energy"}
    url_base = "https://api.nal.usda.gov/fdc/v1/foods/search?query="
    recipe_nlg_df = pd.read_csv(args.input_path)
    ingredients = get_nlg_ingredient_dictionary(recipe_nlg_df)

    for ingredient in tqdm(ingredients):
        if ingredient in ret_df.ingredient.values:
            continue
        ingredient_url = ingredient.replace(" ", "%20")
        url = f"{url_base}{ingredient_url}&api_key={args.api_key}"
        
        response = requests.get(url)
        while response.status_code != 200:
            print(f"Can't get proper response... waiting | status code: {response.status_code}")
            time.sleep(60)
            response = requests.get(url)

        data = response.json()
        ingredient_nutrients = dict(ingredient=ingredient)

        try:
            parsed_nutrients = {nutrient_name: False for nutrient_name in nutrient_ids.values()}
            best_food_data = data['foods'][0]
            ingredient_nutrients['description'] = best_food_data['description']

            for nutrient in best_food_data['foodNutrients']:
                if nutrient['nutrientId'] in nutrient_ids.keys():
                    nutrient_name = nutrient_ids[nutrient['nutrientId']]
                    ingredient_nutrients[nutrient_name] = nutrient['value']
                    ingredient_nutrients[f"{nutrient_name}_unit"] = nutrient['unitName']
                    parsed_nutrients[nutrient_name] = True

            for nutrient_name, was_parsed in parsed_nutrients.items():
                if not was_parsed:
                    ingredient_nutrients[nutrient_name] = np.nan
                    ingredient_nutrients[f"{nutrient_name}_unit"] = np.nan
        except:
            print(f"Something went wrong with {ingredient = }")
            continue

        new_df = pd.DataFrame(data=[ingredient_nutrients], columns=ret_df_columns)
        ret_df = pd.concat([ret_df, new_df])
    
    ret_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main(parse_args())
