## Scripts and presentation for the final milestone

Source of the RoBERTa contrastive model implementation is the work of Witold Sosnowski, Karolina Seweryn, Anna Wróblewska, Piotr Gawrysiak from the paper: 'Revisiting Distance Metric Learning for Few-Shot Natural Language Classification'.

All notebooks were run on the Google Colab virtual environment and may contain paths to the folders in authors private Google Drive.

Data used in the experiments is available at https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Experiments were conducted on the Google colab platform. In order to run them, downloading the dataset and uploading it to the Google disk is necessary. 

There are four main scripts:
* BERT_contrastive_voting_Final - script that implements and tests contrastive learning + voting approach. In the first part, exemplary contrastive architecture is presented. In the second part, it is combined with voting and verified on the validation sample
* Bag_of_Words_experiments - script with implementation and testing of the Bag-of-Words approach. In the “Bag of Words tests” section, we test the implementation and data preprocessing on the IMDB dataset. In the section “BOW final”, we create the embedding that will be used in later parts. In the “Bag-of-Words model with encoder” section, we train the model with created embedding and encoder for contrastive learning. However, this approach does not yield satisfying results. In the “Basic Bag-of-Words model without encoder” section, we make a model with only embedding and MLP for classification. This gives the best results shown in the following sections. In the last section, “Contrastive experiments”, we test other methods for the Bag-of-Words embedding with a contrastive approach.
* RoBERTa_contrastive - in this script, we implemented the RoBERTa contrastive model based on the work of Witold Sosnowski, Karolina Seweryn, Anna Wróblewska, Piotr Gawrysiak from the paper: 'Revisiting Distance Metric Learning for Few-Shot Natural Language Classification'. This script also uses the file RobertaContrastiveModel.py with model creation and data loading functions implemented. In this script, we define hyperparameters for the RoBERTa models and train two versions: RoBERTa-base and RoBERTa-large. Each model is tested for different training dataset sizes, and the following plots are created.
* simcse_voting - script containing model based on embeddings from SimCSE as well as cosine similarities between them and performing classification with k Nearest Neighbors approach. After loading the dataset, simcse_voting_mean() function can be used to compute accuracies and F1 scores for various training dataset sizes for the approach when the label for the new observation is predicted by its average similarity with training samples coming from the two classes (which yielded poor results). Another function, simcse_voting_best_n() can be used for computing accuracies and F1 scores for various training dataset sizes for the final model of this approach, so when label is predicted by majority voting of observation’s k Nearest Neighbors (with k values of 1, 3, 5, 7, and 9)

Also, there are several other files presenting our trials, experiments (and sometimes failures)

