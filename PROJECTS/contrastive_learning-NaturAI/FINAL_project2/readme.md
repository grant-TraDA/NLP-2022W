## Final files and scripts for project 2

Source of the RoBERTa contrastive model implementation is the work of Witold Sosnowski, Karolina Seweryn, Anna Wróblewska, Piotr Gawrysiak from the paper: 'Revisiting Distance Metric Learning for Few-Shot Natural Language Classification'.

All notebooks were run on the Google Colab virtual environment and may contain paths to the folders in authors private Google Drive.

Data used in the experiments is available at https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Experiments were conducted on the Google colab platform. In order to run them, downloading the dataset and uploading it to the Google drive is necessary. 

There are several main scripts:
* BERT_contrastive_voting_functions_tests - script that implements and tests contrastive learning + voting approach. The changed part when compared to project 2 is a new loss function inside training
* BERT_contrastive_sentence_tests2 - scripts that implements a model with reviews split. The most important parts are: 1) review analysis, which prepares training sample with sentences as observations; 2) nested voting approach, which splits test set into sentences and performs a whole training and evaluation of the network
* data_augmentation - scripts performing tests of data augmentation methods and experiments for methods: synonym replacement, embedding-based replacement, summarising text and back-translation.

We store saved models in the following files:
* Siamese_sentence_model.zip 
* https://drive.google.com/drive/folders/1_67V1FQEY1OmxL4F5nbUGR_SdZTaLedc?usp=sharing - all models trained with different augmentation methods (file names denote method_model_dataset used for training). 

Also, there are several other files presenting our trials, experiments (and sometimes failures)


