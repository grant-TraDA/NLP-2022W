# SpoilerDetectionTeam

## Project 2 - Spoiler Detection


## Reproducibility:
1. The code is available in `Final_code` directory.
2. Download the models from https://drive.google.com/drive/folders/1be7d_B6vpnbbhDlcen-4kF0G5Oooe_UK?usp=sharing and put them into `Final_code/checkpoints` directory.
3. Download the glove embeddings https://drive.google.com/file/d/1ht3X_85ScQhiUtUWh-a2eFZAsZW3uYKl/view?usp=sharing and paste the file into `Final_code` directory.
4. Unpack the `Final_code/data/tvtropes_books.zip` archive containing the dataset.

    Make sure that the `Final_code/data` directory contains the following hierarchy:

    ```bash
   (tf) pwesolowski@pop-os:~/Studia/NLP/NLP-2022W/PROJECTS/SpoilerDetectionTeam/project2/Final_code$ ls -lR data
   data:
   total 41720
   drwxrwxr-x 2 pwesolowski pwesolowski     4096 Jan 24 14:12 tvtropes_books
   -rw-rw-r-- 1 pwesolowski pwesolowski 42716655 Jan 24 14:12 tvtropes_books.zip
   
   data/tvtropes_books:
   total 131952
   -rw-rw-r-- 1 pwesolowski pwesolowski  13627872 Jun 21  2020 tvtropes_books-test.json
   -rw-rw-r-- 1 pwesolowski pwesolowski 107887553 Jun 21  2020 tvtropes_books-train.json
   -rw-rw-r-- 1 pwesolowski pwesolowski  13594120 Jun 21  2020 tvtropes_books-val.json
    ```

5. Install the requirements using `Final_code/requirements.txt`. We use Python 3.10.9 if it matters.
6. Reproducing results:
   - LSTM-based models: Run the script `lstm_masking.py`. Possible arguments: `help` (whether to display help),
     `attention` (wheter to use LSTM model featuring attention layer), `train` (whether to train the model). We assume that you load the weights we provide,
     so don't use the `train` argument. Example command: `python lstm_masking.py`, or `python lstm_masking.py attention`.
     The evaluation metrics will be displayed at the end of the script output.
   - BERT-based models: Run the script `bert_models.py`. Possible arguments: `help`, `bert` or `distilbert` (provide one of them), `cased` or `uncased` (provide one of them),
    `train` (whether to train the models). We assume that you load the weights we provide, so don't use `train` argument. Example command: `python bert_models.py bert uncased` or `python bert_models.py distilbert cased`.
     The evaluation metrics will be display at the end of the script output.

In addition, we provide a notebook with exploratory data analysis which was used for the report.
