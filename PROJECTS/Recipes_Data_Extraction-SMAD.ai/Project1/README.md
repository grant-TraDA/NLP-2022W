recipe-data-extraction
==============

Project for NLP 2022 winter course 

Important files
--------------------
1. [EDA](src/notebooks/EDA.ipynb) -- Here, we make an exploratory data analysis on three different datasets:
    * *recipeNLG*
    * *TASTEset*
    * *Food.com downloaded from Kaggle*
2. [BERT training script](scripts/dietary_tags_classification/training_bert.py) -- used to train the language model for our **classification** task.
3. [TF-IDF training script](scripts/dietary_tags_classification/train_tf_idf.py) -- used to train classical ml models on TF-IDF transform from ingredient names
4. [Preprocessing file to create dietary tags dataset](scripts/dietary_tags_classification/preprocess_dietary_tags.py) -- used to preprocess [Food.com](https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags) dataset and convert website tags into dietary tags using `misc/categories_to_tags_mappings.json` mapping.

# Reproducing results
To reproduce our results follow below instructions
## Environment 
1) Prepare python3.9
2) Run `pip install -r requirements.txt`
3) Run `pip install -e .`
   
## Preprocessing
1) Download data from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags)
2) Place data in `dataset`
3) Run `python scripts/dietary_tags_classification/preprocess_dietary_tags.py`

## Training Language model
1) Run `python scripts/dietary_tags_classification/training_language_model.py`

## Training LightGBM on TF-IDF
1) Run `python scripts/dietary_tags_classification/train_tf_idf.py`

### <span style="color:red"> __Warning__</span>
All our training scripts use wandb logging to disable it change `use_wandb_logger` argument in `scripts/dietary_tags_classification/training_config.yaml` file to `False`.