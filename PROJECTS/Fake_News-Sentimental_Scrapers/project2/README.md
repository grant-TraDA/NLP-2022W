# Project 2 - Fake News Detection
We continued working on fake news classification task with the following extensions.
- multi-class classification with the degree of certainty that a piece of news is fake (for example, 0-5 scale)
- datasets with extended input, for example justification
- providing analysis how the choice of inputs influence the models' performance
application of generative models: SentiGAN and CatGAN

The project code is divided into a few different directories with particular architectures tested on two datasets: 
- [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS)
- [CT-FAN](https://zenodo.org/record/4714517)

# Requirements
## BERT
- `Python 3.11` recommended
- GPU support recommended to run the models
- create `conda` environment `nlp-transformers` using the requirements file placed in `BERT/bert_environment.yml` using the following commands

```
conda env create -f BERT/bert_environment.yml -n <your-env-name>

conda activate <your-env-name>
```
All of the code was implemented in separate Jupyter notebooks, which means that each experiment can be presented in details. 

The Google Drive link for the models (they should be placed in `./models/` directory for the evaluation notebooks):

- https://drive.google.com/drive/folders/1nmslj_LOZuXovpGgC7lBa54l3WXflRki?usp=sharing
- the models are quite big (~500 MB)



# In Details
```
├──  BERT  - all notebooks and code for BERT are here
│    └── models  - here the downloaded models should be saved
│    └── *.ipynb  - all of the experiments and preprocessing
|    └── bert_environment.yml  - conda environment dependancies
|    └── bert_utils.py  - tokenisation function for BERT
│
├──  data  
│    └── CT-FAN
|    └── LIAR-PLUS
│
│
├──  LSTM
|
|
├──  CT-FAN_EDA.ipynb  - EDA for CT-FAN dataset
├──  LIAR_PLUS_EDA.ipynb  - EDA for LIAR-PLUS dataset
```


# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.

