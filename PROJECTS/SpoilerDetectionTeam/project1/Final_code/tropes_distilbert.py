import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import urllib.request
import os
import json
import gzip
from transformers import create_optimizer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification
from datasets import Dataset

DATA_PATH = "./data"

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(data):
    return tokenizer(data["text"], truncation=True, padding=True)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf.config.list_physical_devices("GPU")
if model_name == "distilbert-base-uncased":
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, dropout=0.2
    )
elif model_name == "bert-base-uncased":
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, classifier_dropout=0.1
    )


def get_all_sentences(sentences):
    all_sentences = []
    for sentence in sentences:
        all_sentences.append(sentence[1])
    return " ".join(all_sentences)


def get_reviews_from_tropes(filename):
    with open(filename, "rb") as f:
        reviews = []
        for line in f:
            book = json.loads(line)
            reviews.append(
                {
                    "text": get_all_sentences(book["sentences"]),
                    "label": book["has_spoiler"],
                }
            )
        return reviews


def tropes_to_tf(tropes_list):
    tokenized_tropes = Dataset.from_list(tropes_list).map(
        preprocess_function, batched=True
    )
    return model.prepare_tf_dataset(
        tokenized_tropes, shuffle=True, batch_size=32, collate_fn=data_collator
    )


tropes_train = get_reviews_from_tropes(
    f"{DATA_PATH}/tvtropes_books/tvtropes_books-train.json"
)
tropes_test = get_reviews_from_tropes(
    f"{DATA_PATH}/tvtropes_books/tvtropes_books-test.json"
)
tropes_val = get_reviews_from_tropes(
    f"{DATA_PATH}/tvtropes_books/tvtropes_books-val.json"
)

tf_tropes_train = tropes_to_tf(tropes_train)
tf_tropes_val = tropes_to_tf(tropes_val)
tf_tropes_test = tropes_to_tf(tropes_test)


batch_size = 32

num_epochs = 3

batches_per_epoch = len(tropes_train) // batch_size

total_train_steps = int(batches_per_epoch * num_epochs)

optimizer, schedule = create_optimizer(
    init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps
)


class BalancedSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name="balanced_sparse_categorical_accuracy", dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)


model.compile(
    optimizer=optimizer, metrics=["accuracy", BalancedSparseCategoricalAccuracy()]
)

checkpoint_name = f"./checkpoints/best_val_tropes_{model_name}"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_name,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

csv_logger = tf.keras.callbacks.CSVLogger(
    f"tropes_fit_log_{model_name}.csv", append=True, separator=";"
)


from timeit import default_timer as timer

start = timer()
# model.fit(
#     x=tf_tropes_train,
#     validation_data=tf_tropes_val,
#     epochs=3,
#     callbacks=[model_checkpoint_callback, csv_logger],
# )
end = timer()
print("Number of seconds spent fitting")
print(end - start)
print("------")


model.load_weights(checkpoint_name)

start = timer()
model.evaluate(tf_tropes_test)
end = timer()
print("Number of seconds spent evaluating")
print(end - start)
print("------")


auc = tf.keras.metrics.AUC()
acc = tf.keras.metrics.Accuracy()

fp = tf.keras.metrics.FalsePositives()
fn = tf.keras.metrics.FalseNegatives()
tp = tf.keras.metrics.TruePositives()
tn = tf.keras.metrics.TrueNegatives()

for batch_data, batch_labels in tqdm(tf_tropes_test):
    preds = tf.nn.softmax(model(batch_data)[0])[:, 1]
    auc.update_state(batch_labels, preds)
    acc.update_state(batch_labels, preds >= 0.5)

    fp.update_state(batch_labels, preds >= 0.5)
    fn.update_state(batch_labels, preds >= 0.5)
    tp.update_state(batch_labels, preds >= 0.5)
    tn.update_state(batch_labels, preds >= 0.5)
