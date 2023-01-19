MOP E-COMMERCE Probing tasks
============
Authors: Paweł Golik | Mateusz Jastrzębiowski | Aleksandra Muszkowska (MOP Team)
---

 The project was created during the Natural Language Processing (NLP) course at Warsaw University of Technology, Faculty of Mathematics and Information Science (2022/23 academic year).

---

## About

 Probing is a tool for investigating embedded spaces created using BERT-like transformers. They involve building new classifiers trained on embeddings to see if we can verify the hypothesis about the embedded space by acquiring good accuracy of the probing classifier on the embeddings.

---

## Related works

### Probing tasks
If you want to learn more about probing tasks, please read the following papers:
 - general introduction of probing tasks [Sahin et al. 2020]
 - probing tasks for visual-semantic case, this paper was our inspiration that probing tasks can be more intricate [Lindstrom et al. 2020]

### Cross-encoders vs. Bi-encoders architecture
Cross-encoders [Wolf et al., 2019] usually yield the best results, but they are not suitable for retrieving of the embeddings.
On the other hand, Bi-encoders [Mazare et al. 2018] are slightly faster and less accurate than cross-encoders, but they produce an explicit embedding for each of the sentence in pair.
Another advantage is that the fine-tuning is based on comparing the similarities of those two sentence embeddings.

The SentenceTrasfomers library documentation provides a great explanation of cross/bi-encoders, as well as a sample code for using them.


### WDC dataset for e-commerce product matching
[Mozdzonek et al.2022] describes the problem and uses state-of-the-art cross-encoder to solve the problem

--- 

## Libraries
- [Hugging Face](https://huggingface.co/) [Wolf et al.2019]: A great library with many pretrained language models, such as: `bert-base-cased` or `xml-roberta-base`.
- [SentenceTransformers](https://www.sbert.net/) [Reimers et al.2019]: A library designed for easy computing of sentence embeddings using Hugging Face transformers under the hood.
It also provides ready-to-use functions for their fine-tunning.

---

## Dataset
**Web Data Commons - Training Dataset and Gold Standard for Large-Scale Product Matching** - the dataset consists of pairs of offers grouped into four categories: `Computers`, `Cameras`, `Watches`, `Shoes`. Each pair of offers is either a `positive` pair (both offers regard the same product) or a `negative` pair (two different products). Please, note that offers within negative pairs regard different products but still, of the same product category (e.g. two different cameras) only.

To learn more about the dataset visit the website: [WDC dataset](http://webdatacommons.org/largescaleproductcorpus/).

---

## Project #1 and Project #2 scopes

 It is divided into two separate projects (Project #1 and Project #2).

 Project #1 covers:
 - fine-tuning of the `bert-base-cased` model on the `Cameras medium` WDC dataset (e-commerce product matching problem) 
 - calculating embeddings for each offer using the fine-tuned model
 - creating probing tasks to explain the obtained embedded space:
    - **Common words**: the goal was to build a classifier, that based on the embedding of an offer, tried to predict whether the offer contains at least one of the common words
    ('camera', 'digital', 'lens');
    - **Brand name**: the goal was to build a classifier, that based on the embedding of an offer, tried to predict whether the offer contains a brand name (e.g., 'Canon') or not;
    - **Length of sentences**: the goal was to build a classifier, that based on the embedding of an offer, tried to predict the length of the input (string containing the offer);
    - **The Levenshtein distance**: the goal was to build a classifier, that based on the pair of embeddings, tried to predict the Levenshtein distance score calculated from the two strings (each representing one offer from the pair). We discretized the target variable into 5 classes ('similar', 'slightly similar', 'neutral', 'hardly similar', 'not similar').

 Project #2 covers:
 - fine-tuning of the `xml-roberta-base` model on the `Computers medium` WDC dataset (e-commerce product matching problem)
 - calculating embeddings for each offer using the fine-tuned model
 - calculating embeddings using the model before fine-tuning to analyze the influence of fine-tunning on the space
 - performing probing tasks from the Project #1 on embeddings obtained both before and after fine-tunning
 - designing new probing tasks and performing them on embeddings obtained both before and after fine-tunning
 - applying the probing tasks based on string similarity to a "natural" dataset (without specialized words such as: names of brands, models)

---

## Setup
1. Clone this repo to your desktop.

2. Visit `config.py` to change the `BASE_PATH` constant - provide the absolute path to the `NLP-2022W-MOP` (e.g., `r"C:/Users/user/Desktop/repo/"`).
Change other config parameters if needed.

3. Create `anaconda` environment and install all dependencies (listed in the `requirements.txt` file): `conda create --name <env_name> --file requirements.txt`

---

## Usage

### Notebooks
You may want to take a look at our notebooks. You can find there our work.
The notebooks reside in the directories `project1_notebooks` (for Project #1) and `project2_notebooks` (for Project #2).

Project #1 notebooks:
- `finetuning_embedding_extraction.ipynb`: this notebook is meant to be run on Google Colab environment. It was used for fine-tuning of the selected model, embedding extraction and saving results.
- `probing_tasks.ipynb`: in this notebook you can find our probing tasks
- `export_tsv.ipynb`: notebook used for converting the embeddings into a .tsv file for visualization of the embedded space at https://projector.tensorflow.org/.

Project #2 notebooks:
TODO (Project #2 still in progress)


### Source code
All functions/classes used throughout the notebooks can be found in the `source` directory to provide easy reusing of the code in future notebooks/scripts.


### Models
To ensure reproducibility of experiments and results, we added fine-tuned models in the directories `project1_models` (for Project #1) and `project2_models` (for Project #2).
Each model is placed in its own subdirectory (e.g., `project1_models/bert-base-cased`) accompanied by the `info.txt` file, in which we provided additional information about fine-tuning (`hyperparameters`, used functions etc., exact dataset). The model binaries and additional configuration files (produced by the SentenceTransformers library - refer to doc) were packed into a zip archive. They are available via the url provided in  the `model_url.txt` file (due to the GitHub's limit for uploading large files).


### Scripts
Scripts are located in the directory: `scripts`

1. `train.py` - script that can be used for transformer fine-tunning: 
`python train.py --outdir ./output --hugging_face_model  bert-base-cased --batch_size 16 --dataset_type cameras --dataset_size medium --num_epochs 200`

2. `extract_embeddings.py` - extracts embeddings for a given dataset using a given model:
`python extract_embeddings.py --outdir ./output --model_inputdir ./models/bert-base-cased --dataset_type cameras --dataset_size medium`

3. `probe.py` - performs probing tasks using given embeddings and the corresponding dataset: 
`python probe.py --outdir ./output --model_inputdir ./models/bert-base-cased --dataset_type cameras --dataset_size medium`

You can use the `--help` command to learn about possible parameters passed to the scripts.

### Outputs

In the directories `project1_output` (for Project #1) and `project2_output` (for Project #2), we store all output files (such as embeddings, plots, images etc.)


### Proof of Concept

The directories `POC` (for Project #1) and `POC2` (for Project #2) contain the source code and notebooks presented during the PoC classes.

### Dataset

The directory `datasets` contains data used in the project.

### Report and Presentation

In directory `project1_output/Final_report_and_presentation` the final Report pdf file, final tex file and presentation is provided.

---


## Reproducibility 
As we described in Section `Usage`, we attached fine-tuned models with files describing the fine-tunning process.
We also uploaded scripts for fine-tunning, embedding extraction and probing tasks.

---


## Bibliography



[Lindstrom et al. 2020]  Lindstrom, Adam & Bj ̈orklund,
Johanna & Bensch, Suna & Drewes, Frank. 2021.
Probing Multimodal Embeddings for Linguistic
Properties: the Visual-Semantic Case. In Proceed-
ings of the 28th International Conference on Com-
putational Linguistics, pages 730–744, Barcelona,
Spain (Online). International Committee on Compu-
tational Linguistics.

[Sahin et al. 2020] Sahin, G ̈ozde G ̈ul and Vania, Clara
and Kuznetsov, Ilia and Gurevych, Iryna 2020.
LINSPECTOR: Multilingual Probing Tasks for
Word Representations. Computational Linguistics.
46, 335-385 (2020,6), https://doi.org/10.1162/col

[Mozdzonek et al.2022]  Mozdzonek, Michał &
Wr ́oblewska, Anna & Tkachuk, Sergiy & Łukasik,
Szymon 2022. Multilingual Transformers for
Product Matching – Experiments and a New
Benchmark in Polish.. 1-8. 10.1109/FUZZ-
IEEE55066.2022.9882843.

[Wolf et al.2019] Wolf, Thomas & Sanh, Victor &
Chaumond, Julien & Delangue, Clement. 2019.
Transfertransfo: A transfer learning approach for
neural network-based conversational agents.

[Wolf et al.2019] Wolf, Thomas & Debut, Lysandre &
Sanh, Victor & Chaumond, Julien & Delangue, Clement &
Moi, Anthony & Cistac, Pierric & Rault, Tim &
Louf, Remi & Funtowicz, Morgan and others. 2019
Huggingface's transformers: State-of-the-art natural language processing

[Reimers et al.2019] Reimers, Nils & Gurevych, Iryna. 2019
Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.


[Mazare et al.2018] Mazare, Pierre-Emmanuel &
Humeau, Samuel & Raison, Martin & Bordes,
Antoine. 2018. Training millions of personalized
dialogue agents.


