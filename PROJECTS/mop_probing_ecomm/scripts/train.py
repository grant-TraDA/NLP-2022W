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




@click.command()

# Required.
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--model_inputdir', help='Directory where model to be fine-tuned resides', metavar='DIR', required=False)
@click.option('--hugging_face_model', help='Name of the Hugging Face transformer', metavar='STR', required=False)
@click.option('--batch_size', help='Size of batches', metavar='INT', required=False, default=16, show_default=True)
@click.option('--dataset_type', help='WDC dataset category', metavar='STR', required=True, default='cameras', show_default=True)
@click.option('--dataset_size', help='WDC dataset size', metavar='STR', required=True, default='medium', show_default=True)
@click.option('--num_epochs', help='Number of epochs', metavar='INT', required=True, default=200, show_default=True)

def main(outdir, model_inputdir, hugging_face_model, batch_size, dataset_type, dataset_size, num_epochs):
    model_save_path = outdir
    model_name = hugging_face_model

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) 

    train_samples = get_wdc_dataset(dataset_type, dataset_size, is_train=True, features_to_concat=['title'])
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    test_samples = get_wdc_dataset(dataset_type, dataset_size, is_train=False, features_to_concat=['title'])
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='test_evaluation')

    # Configure the training. 
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model and save it
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)


if __name__ == "__main__":
    main()
