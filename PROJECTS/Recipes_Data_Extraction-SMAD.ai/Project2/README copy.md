# LLM-for-recipes
## Environment setup
To setup environment follow below steps
* install python 3.9
* install packages with `pip install -r requirements.txt`
* paste your OpenAI API key into `openai_key.txt` file
## Reproducing results
### Data
To reproduce our results first you need to prepare data

1) Put [Kaggle](https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags) dataset into folder `dataset/`.
2) run `python scripts/preprocessing/preprocess_dietary_tags.py` 
3) run `python scripts/preprocessing/sample_classification_df.py`

### API calls
To get responses from GPT-3 model run below commands
1) `python scripts/getting_model_outputs/ingredients_extraction.py`
2) `python scripts/getting_model_outputs/specified_dietary_tags.py`
3) `python scripts/getting_model_outputs/unspecified_dietary_tags.py`

### Calculating metrics
Calculation of metrics and plots can be found in `scripts/evaluation/evaluate_model_outputs.py`
   

### Testing different prompts 
We also tried to test some other prompts ex. NER or Ingredients replacement it can be found in notebook `notebooks/tweaking_with_prompts.ipynb`

## Deliverable table
| Deliverable name | Deliverable justification | 
| ---------------- | ------------------------- |
| Report | Report can be found on this github at file `NLP_project2_final_report.pdf` we also provide link to our overleaf project at `final_report_overleaf_link.txt`|
| Clean code | Our code is structured and contains comments explaining how it works. Most important files are explained in github README |
| Reproducibility | In github README we explain how to reproduce our results step by step following those steps is sufficient to get exact results as we did |
| Contributions | Our projects contribution is testing how well LLMs work on recipe domain. With our tests we were able to see when these models fail to work and when they thrive and can be really helpful |  