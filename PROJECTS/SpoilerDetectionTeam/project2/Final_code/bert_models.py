DATA_PATH = r"./data/tvtropes_books"

import tensorflow as tf
import json
from pathlib import Path
import sys

args = set(sys.argv[1:])

if "help" in args:
    print(
        "Make sure to change DATA_PATH at top of the script.\nPossible arguments: distilbert/bert, cased/uncased, train (optional), help (optional)")
    exit()

if "distilbert" in args and "bert" in args:
    print("Provide either distilbert or bert")
    exit()

if "cased" in args and "uncased" in args:
    print("Provide either cased or uncased")
    exit()

print("Running Spoiler detection script")
print(args)
CASED = "cased" in args
DISTILBERT = "distilbert" in args
print(f"CASED={CASED}, DISTILBERT={DISTILBERT}")
print()

# Reading the data
for fname in (f"tvtropes_books-{suffix}.json" for suffix in ["train", "test", "val"]):
    assert (Path(DATA_PATH) / fname).is_file(), f"File {fname} not found"

train_list, val_list, test_list = [], [], []
for path, which in ((Path(DATA_PATH) / f"tvtropes_books-{suffix}.json", suffix) for suffix in ["train", "test", "val"]):
    with open(path, "r") as f:
        for line in f:
            globals()[f"{which}_list"].append(json.loads(line))

print("Lengths of the data read: ", len(train_list), len(val_list), len(test_list))

# Preprocessing the data, namely for each token we want to assign specific label
DIM = 512


def in_range(interval_1, interval_2):
    assert interval_1[0] <= interval_1[1] and interval_2[0] <= interval_2[1]
    return interval_1[0] >= interval_2[0] and interval_1[1] <= interval_2[1]


import re


def split_with_position(str_):
    word_pos_list = []
    for m in re.finditer(r'\S+', str_):
        pos, word = m.span(), m.group()
        word_pos_list.append((word, pos))
    return word_pos_list


# The below function splits the texts into words and takes into account their positions in the original text
# It's needed because of how tv_tropes_books is annotated
def text_to_word_sequence(
        input_text,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=not CASED,
        split=" ",
        return_pos=True
):
    if lower:
        input_text = input_text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    input_text = input_text.translate(translate_map)

    if return_pos:
        return tuple(zip(*[(word, pos) for word, pos in split_with_position(input_text) if word]))
    else:
        return tuple(word for word, pos in split_with_position(input_text) if word)


# Preparing the appropriate Datasets...
from datasets import Dataset


def prepare_dataset(data_list):
    X_list = []
    y_list = []
    for data in data_list:
        if data["has_spoiler"]:
            sentence_data = data["sentences"]
            i = 0
            while i < len(sentence_data):
                # i points to the sentence to process
                input_words_list = []
                input_labels_list = []
                cur_words_count = 0

                while i < len(sentence_data):
                    next_sentence_words, next_sentence_word_positions = text_to_word_sequence(sentence_data[i][1])
                    if cur_words_count + len(next_sentence_words) > DIM:
                        if len(next_sentence_words) > DIM:
                            i += 1
                        break
                    cur_words_count += len(next_sentence_words)
                    input_words_list.extend(next_sentence_words)
                    input_labels_list.extend(
                        int(any(in_range(pos, spoiler_boundary) for spoiler_boundary in sentence_data[i][2])) for pos in
                        next_sentence_word_positions)
                    i += 1

                if input_words_list:
                    X_list.append(input_words_list)
                    y_list.append(input_labels_list)
    return Dataset.from_list([{"tokens": x, "labels": y} for x, y in zip(X_list, y_list)])


ds_train = prepare_dataset(train_list)
ds_val = prepare_dataset(val_list)
ds_test = prepare_dataset(test_list)

from transformers import AutoTokenizer

if DISTILBERT:
    model_checkpoint = "distilbert-base-cased" if CASED else "distilbert-base-uncased"
else:
    model_checkpoint = "bert-base-cased" if CASED else "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# The below functions are needed because BERT tokenizer may produce some additional tokens,
# refer to https://huggingface.co/course/chapter7/2?fw=tf for further details
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    assert len(new_labels) == len(examples["tokens"])
    return tokenized_inputs


tokenized_ds_train = ds_train.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns="tokens"
)
tokenized_ds_val = ds_val.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns="tokens"
)
tokenized_ds_test = ds_test.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns="tokens"
)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, return_tensors="tf"
)


# Creating TF datasets
def create_tf_dataset(tokenized_ds, shuffle):
    if DISTILBERT:
        return tokenized_ds.to_tf_dataset(
            columns=["attention_mask", "input_ids", "labels"],
            collate_fn=data_collator,
            shuffle=shuffle,
            batch_size=32,
            prefetch=True
        )
    else:
        return tokenized_ds.to_tf_dataset(
            columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
            collate_fn=data_collator,
            shuffle=shuffle,
            batch_size=32,
            prefetch=True
        )


tf_ds_train = create_tf_dataset(tokenized_ds_train, True)
tf_ds_val = create_tf_dataset(tokenized_ds_val, False)
tf_ds_test = create_tf_dataset(tokenized_ds_test, False)

id2label = {i: label for i, label in enumerate(["non-spoiler", "spoiler"])}
label2id = {v: k for k, v in id2label.items()}

from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

assert model.config.num_labels == 2


# Custom accuracy metrics. I had to define them due to -100 labels corresponding to trash tokens (padding, etc.)
# In addition, each output has 2 probabilities (class 0 and 1). Therefore, traditional Binary metrics would not work
class CustomBinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        self.metric: tf.keras.metrics.Metric = tf.keras.metrics.BinaryAccuracy(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        weights = tf.cast((y_true != -100), tf.float32)
        y_pred = y_pred[..., 1]
        y_pred = tf.math.greater(y_pred, 0.5)
        self.metric.update_state(y_true, y_pred, sample_weight=weights)

    def reset_state(self):
        self.metric.reset_state()

    def result(self):
        return self.metric.result()


class JaccardSimilarity(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        self.metric: tf.keras.metrics.Metric = tf.keras.metrics.IoU(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        weights = tf.cast((y_true != -100), tf.float32)
        y_true = tf.cast(y_true > 0, y_true.dtype) * y_true  # Jaccard similarity sie wywala, gdy dajemy ujemne labele
        y_pred = y_pred[..., 1]
        y_pred = tf.math.greater(y_pred, 0.5)
        self.metric.update_state(y_true, y_pred, sample_weight=weights)

    def reset_state(self):
        self.metric.reset_state()

    def result(self):
        return self.metric.result()


# Podobienstwo Jaccarda gdy ground truth dla danego labela jest zbiorem pustym będzie 0. Jednak średnie podobieństwo poprawnie ignoruje taki wynik.


class CustomRecall(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        self.metric: tf.keras.metrics.Metric = tf.keras.metrics.Recall(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        weights = tf.cast((y_true != -100), tf.float32)
        y_pred = y_pred[..., 1]
        y_pred = tf.math.greater(y_pred, 0.5)
        self.metric.update_state(y_true, y_pred, sample_weight=weights)

    def reset_state(self):
        self.metric.reset_state()

    def result(self):
        return self.metric.result()


class CustomPrecision(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        self.metric: tf.keras.metrics.Metric = tf.keras.metrics.Precision(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        weights = tf.cast((y_true != -100), tf.float32)
        y_pred = y_pred[..., 1]
        y_pred = tf.math.greater(y_pred, 0.5)
        self.metric.update_state(y_true, y_pred, sample_weight=weights)

    def reset_state(self):
        self.metric.reset_state()

    def result(self):
        return self.metric.result()


from transformers import create_optimizer

num_epochs = 3
num_train_steps = len(tf_ds_train) * num_epochs

optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(
    optimizer=optimizer,
    metrics=[
        CustomBinaryAccuracy("custom_accuracy"),
        CustomRecall("recall"),
        CustomPrecision("precision"),
        JaccardSimilarity('jaccard_nonspoilers', num_classes=2, target_class_ids=[0]),
        JaccardSimilarity('jaccard_spoilers', num_classes=2, target_class_ids=[1]),
        JaccardSimilarity('mean_jaccard', num_classes=2, target_class_ids=[0, 1]),
    ],
)

checkpoint_name = f"./checkpoints/{model_checkpoint}"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_name,
    save_weights_only=True,
    monitor="val_custom_accuracy",
    mode="max",
    save_best_only=True,
)

csv_logger = tf.keras.callbacks.CSVLogger(
    f"final_fit_log_{model_checkpoint}.csv", append=True, separator=";"
)

if "train" in args:
    model.fit(
        tf_ds_train,
        validation_data=tf_ds_val,
        epochs=num_epochs,
        callbacks=[model_checkpoint_callback, csv_logger],
        verbose=2
    )

print("Final metrics")

model.load_weights(checkpoint_name)
model.evaluate(tf_ds_test, verbose=2)
