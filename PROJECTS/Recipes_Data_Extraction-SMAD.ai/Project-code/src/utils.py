import ast
from itertools import chain

import pandas as pd
import numpy as np


def get_nlg_ingredient_dictionary(df):
    ingredients = df.NER.apply(lambda x: ast.literal_eval(x)).values.tolist()
    ingredients = list(chain(*ingredients))
    ingredients = [ingredient.lower() for ingredient in ingredients]
    ingredients = set(ingredients)

    return ingredients
