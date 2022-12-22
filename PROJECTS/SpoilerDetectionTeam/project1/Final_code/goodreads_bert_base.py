import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
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

urls = [
    "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-train.json.gz",
    "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-val.json.gz",
    "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-test.json.gz",
]

FORCE_REDOWNLOAD = False

goodreads_train = []
goodreads_val = []
goodreads_test = []

for goodreads_list, url in zip([goodreads_train, goodreads_val, goodreads_test], urls):
    file = f"{DATA_PATH}/goodreads/{url.rsplit('/', 1)[-1]}"
    if not os.path.exists(file) or FORCE_REDOWNLOAD:
        urllib.request.urlretrieve(url, file)

    with gzip.open(file, "rb") as f:
        for line in tqdm(f):
            goodreads_list.append(json.loads(line))


model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)


def preprocess_function(data):
    return tokenizer(data["text"], truncation=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
if model_name == "distilbert-base-uncased":
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, dropout=0.2
    )
elif model_name == "bert-base-uncased":
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, classifier_dropout=0.1
    )


def create_tf_dataset(goodreads_list):
    tokenized_goodreads = Dataset.from_list(
        [
            {
                "text": "".join(y[1] for y in x["review_sentences"]),
                "label": x["has_spoiler"],
            }
            for x in goodreads_list
        ]
    ).map(preprocess_function, batched=True)
    return model.prepare_tf_dataset(
        tokenized_goodreads, shuffle=True, batch_size=32, collate_fn=data_collator
    )


tf_goodreads_train = create_tf_dataset(goodreads_train)
tf_goodreads_val = create_tf_dataset(goodreads_val)
tf_goodreads_test = create_tf_dataset(goodreads_test)


batch_size = 32

num_epochs = 3

batches_per_epoch = len(goodreads_train) // batch_size

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

checkpoint_name = f"./checkpoints/best_val_goodreads_{model_name}"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_name,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

csv_logger = tf.keras.callbacks.CSVLogger(
    f"goodreads_fit_log_{model_name}.csv", append=True, separator=";"
)


from timeit import default_timer as timer

start = timer()
# model.fit(
#     x=tf_goodreads_train,
#     validation_data=tf_goodreads_val,
#     epochs=3,
#     callbacks=[model_checkpoint_callback, csv_logger],
# )
end = timer()
print("Number of seconds spent fitting")
print(end - start)
print("------")


model.load_weights(checkpoint_name)

start = timer()
model.evaluate(tf_goodreads_test)
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

for batch_data, batch_labels in tqdm(tf_goodreads_test):
    preds = tf.nn.softmax(model(batch_data)[0])[:, 1]
    auc.update_state(batch_labels, preds)
    acc.update_state(batch_labels, preds >= 0.5)

    fp.update_state(batch_labels, preds >= 0.5)
    fn.update_state(batch_labels, preds >= 0.5)
    tp.update_state(batch_labels, preds >= 0.5)
    tn.update_state(batch_labels, preds >= 0.5)

print("Non whole goodreads")
print(f"AUC={auc.result().numpy()}")
print(f"ACC={acc.result().numpy()}")
print(f"FP={fp.result().numpy()}")
print(f"FN={fn.result().numpy()}")
print(f"TP={tp.result().numpy()}")
print(f"TN={tn.result().numpy()}")

print("Reading whole goodreads")

from itertools import chain

exclude_reviews = set(x["review_id"] for x in chain(goodreads_train, goodreads_test))
len(exclude_reviews)
whole_goodreads_test = []
file = f"{DATA_PATH}/goodreads/goodreads_review_spoiler.json"

with open(file, "r") as f:
    for line in tqdm(f):
        as_json = json.loads(line)
        if as_json["review_id"] not in exclude_reviews:
            whole_goodreads_test.append(as_json)
tf_whole_goodreads_test = create_tf_dataset(whole_goodreads_test)
del whole_goodreads_test

print("Creating new metrics")
auc = tf.keras.metrics.AUC()
acc = tf.keras.metrics.Accuracy()

fp = tf.keras.metrics.FalsePositives()
fn = tf.keras.metrics.FalseNegatives()
tp = tf.keras.metrics.TruePositives()
tn = tf.keras.metrics.TrueNegatives()

for batch_data, batch_labels in tqdm(tf_whole_goodreads_test):
    preds = tf.nn.softmax(model(batch_data)[0])[:, 1]
    auc.update_state(batch_labels, preds)
    acc.update_state(batch_labels, preds >= 0.5)

    fp.update_state(batch_labels, preds >= 0.5)
    fn.update_state(batch_labels, preds >= 0.5)
    tp.update_state(batch_labels, preds >= 0.5)
    tn.update_state(batch_labels, preds >= 0.5)

print("Whole goodreads")
print(f"AUC={auc.result().numpy()}")
print(f"ACC={acc.result().numpy()}")
print(f"FP={fp.result().numpy()}")
print(f"FN={fn.result().numpy()}")
print(f"TP={tp.result().numpy()}")
print(f"TN={tn.result().numpy()}")
