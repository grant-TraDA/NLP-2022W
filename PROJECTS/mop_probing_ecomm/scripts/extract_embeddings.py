import click
import logging
import datetime
import math

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# source code imports
import os, sys
current_dir = os.path.abspath('')
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from source.load_data.wdc.load_wdc_dataset import get_wdc_dataset
from source.emb_prep_res.compute_save_emb import get_embedding_records, get_embeddings_pairs, create_csv_file





@click.command()

# Required.
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--model_inputdir', help='Directory where model to be fine-tuned resides', metavar='DIR', required=False)
@click.option('--batch_size', help='Size of batches', metavar='INT', required=False, default=16, show_default=True)
@click.option('--dataset_type', help='WDC dataset category', metavar='STR', required=True, default='cameras', show_default=True)
@click.option('--dataset_size', help='WDC dataset size', metavar='STR', required=True, default='medium', show_default=True)

def main(outdir, model_inputdir, batch_size, dataset_type, dataset_size):
    
    embeddings_file_path_train = f'{outdir}/train_embeddings.csv'
    embeddings_file_path_test = f'{outdir}/test_embeddings.csv'

    field_names = ['id', 'embedding']
    model_save_path = model_inputdir

    model = SentenceTransformer(model_save_path)
    train_samples = get_wdc_dataset(dataset_type, dataset_size, is_train=True, features_to_concat=['title'])
    test_samples = get_wdc_dataset(dataset_type, dataset_size, is_train=False, features_to_concat=['title'])
   
    train_embeddings_1, train_embeddings_2 = get_embeddings_pairs(train_samples, model, batch_size=batch_size)
    test_embeddings_1, test_embeddings_2 = get_embeddings_pairs(test_samples, model, batch_size=batch_size)

    train_records = get_embedding_records(train_samples, train_embeddings_1, train_embeddings_2)
    test_records = get_embedding_records(test_samples, test_embeddings_1, test_embeddings_2)

    create_csv_file(embeddings_file_path_train, field_names, train_records)
    create_csv_file(embeddings_file_path_test, field_names, test_records)

if __name__ == "__main__":
    main() 