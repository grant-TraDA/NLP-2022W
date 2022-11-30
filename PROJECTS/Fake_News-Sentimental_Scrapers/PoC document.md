---
title: NLP - Fake News
created: '2022-11-24T23:34:46.865Z'
modified: '2022-11-25T00:52:16.373Z'
---

# NLP - Fake News

## 25/11/2022 - __Proof of Concept__

### Unique Contributions and Relations with Research

The detailed research inspired us to test architectures different in complexity and ideas. We want to test different groups of architectures stated in the papers:
- CNNs, RNNs
- mixed models
- transformers
- GANs

In general, fake news detection tasks and natural language models are new to us, and we want to experiment with different methods on a not-that-difficult task.

Our unique contirbutions:
- analysing how different preprocessing techniques influence the results of different architectures
- testing whether keeping some stop words like _no_, _yes_, etc. influence the training process and predictions
- check whether a specific model trained on one fake news dataset generalises to other datasets of a similar topic or different topics. Is the question - do models usually overfit, or can they recognise patterns and manipulations in fake news articles?

### Answering Reviews

#### Too broad approach - difficult to examine all listed methodologies

We agree on this point and we will not try to test every architecture mentioned. Only ones that seem interesting to us.

#### No novelty 

Already addressed in the first section. We did not make our plans clear at the point of the presentation.

#### Little diversity in datasets

We are still looking for some more datasets to work with. Longer articles will be difficult to train without additional resources. We will try to test our models on benchmarks from _paperswithcode.com_.

#### State-of-the-art analysis was too short

We do not agree on this opinion. There were a lot of research mentioned. 

#### Not cited datasets and models

All necessary citations are presented in the report, which we mentioned during the presentation. 

#### Why is it better that the datasets are balanced?

It is more important to detect fake news than to confirm that the news is indeed true. Unbalanced dataset with small number of negative cases will result in good accuracy, but the ability to detect fake news in test datasets will be poor without adjustments.

#### Different models to test, but no interpretation with tools etc.

We do not understand this point. Is it about technologies?

### EDA, preliminary models, and presenting achievements

All the achievements will be presented in jupyter notebooks.

### Contributions

We all worked on different groups of models that we analysed for the first presentations and implemented a preliminary notebooks with some training. 

Marcel prepared the first exploratory data analysis, and Arkadiusz and Mateusz worked on different preprocessing techniques.


