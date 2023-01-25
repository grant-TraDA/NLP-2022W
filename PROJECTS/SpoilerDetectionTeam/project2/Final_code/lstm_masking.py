DATA_PATH = r"./data/tvtropes_books"

import json
from pathlib import Path
import sys

args = set(sys.argv[1:])

if "help" in args:
    print(
        "Make sure to change DATA_PATH at top of the script.\nPossible arguments: attention (optional), train (optional), help (optional)")
    exit()

print("Running Spoiler detection script")
print(args)
ATTENTION = "attention" in args
TRAIN = "train" in args
print(f"ATTENTION={ATTENTION}, TRAIN={TRAIN}")
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

import re


# The following function splits strings into words and returns also their positions (note that the dataset contains
# annotated spoiler indices)
def split_with_position(str_):
    word_pos_list = []
    for m in re.finditer(r'\S+', str_):
        pos, word = m.span(), m.group()
        word_pos_list.append((word, pos))
    return word_pos_list

# The following function is used in the preprocessing
def text_to_word_sequence(
        input_text,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
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


DIM = 512

# The following function is useful to check wheter the corresponding word is annotated as spoiler or not
def in_range(interval_1, interval_2):
    assert interval_1[0] <= interval_1[1] and interval_2[0] <= interval_2[1]
    return interval_1[0] >= interval_2[0] and interval_1[1] <= interval_2[1]


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
                        any(in_range(pos, spoiler_boundary) for spoiler_boundary in sentence_data[i][2]) for pos in
                        next_sentence_word_positions)
                    i += 1

                if input_words_list:
                    X_list.append(input_words_list)
                    y_list.append(input_labels_list)
    X_list = [" ".join(s) for s in X_list]
    return X_list, y_list


X_train_list, y_train_list = prepare_dataset(train_list)
X_val_list, y_val_list = prepare_dataset(val_list)
X_test_list, y_test_list = prepare_dataset(test_list)

from tensorflow.keras.preprocessing.text import Tokenizer

t = Tokenizer()
# Preprocessed_reviews contains all the cleaned reviews.
t.fit_on_texts(X_train_list)

vocab_size = len(t.word_index) + 1

from tensorflow.keras.preprocessing import sequence
import numpy as np

# Preparing the datasets
X_train = sequence.pad_sequences(t.texts_to_sequences(X_train_list), maxlen=DIM, padding='post')
y_train = np.expand_dims(sequence.pad_sequences(y_train_list, maxlen=DIM, padding='post'), axis=-1)

X_val = sequence.pad_sequences(t.texts_to_sequences(X_val_list), maxlen=DIM, padding='post')
y_val = np.expand_dims(sequence.pad_sequences(y_val_list, maxlen=DIM, padding='post'), axis=-1)

X_test = sequence.pad_sequences(t.texts_to_sequences(X_test_list), maxlen=DIM, padding='post')
y_test = np.expand_dims(sequence.pad_sequences(y_test_list, maxlen=DIM, padding='post'), axis=-1)

embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    EMBEDDING_DIM = coefs.size

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((X_train.max() + 1, EMBEDDING_DIM))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Bidirectional, Attention, Dropout


# Binary accuracy dziala dobrze, bo testowałem - PW

# Custom JaccardSimilarity wrapper
class JaccardSimilarity(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        self.metric: tf.keras.metrics.Metric = tf.keras.metrics.IoU(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.greater(y_pred, 0.5)
        self.metric.update_state(y_true, y_pred,
                                 sample_weight=sample_weight)  # Dzięki temu, że tu mamy sample_weight, to metryka wspiera masking

    def reset_state(self):
        self.metric.reset_state()

    def result(self):
        return self.metric.result()


# Podobienstwo Jaccarda gdy ground truth dla danego labela jest zbiorem pustym będzie 0. Jednak średnie podobieństwo poprawnie ignoruje taki wynik.


np.random.seed(0)
tf.random.set_seed(1)

HIDDEN_DIM = 256
DIM = 512

# Model definition
if ATTENTION:
    inputs = Input(shape=(DIM,))
    x = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, mask_zero=True)(inputs)

    x = Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True))(x)

    x = Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True))(x)

    x = Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True))(x)

    x = Attention()([x, x, x])

    x = Dropout(0.1)(x)

    x = Dense(1, activation='sigmoid')(x)

    outputs = x
else:
    inputs = Input(shape=(DIM,))
    x = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, mask_zero=True)(inputs)

    x = Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True))(x)

    x = Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True))(x)

    x = Dense(1, activation='sigmoid')(x)

    outputs = x

model = Model(inputs, outputs)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[
                  'binary_accuracy',
                  tf.keras.metrics.Recall(),
                  tf.keras.metrics.Precision(),
                  JaccardSimilarity('jaccard_nonspoilers', num_classes=2, target_class_ids=[0]),
                  JaccardSimilarity('jaccard_spoilers', num_classes=2, target_class_ids=[1]),
                  JaccardSimilarity('mean_jaccard', num_classes=2, target_class_ids=[0, 1])
              ],
              optimizer='adam')
model.summary()

checkpoint_name = f"./checkpoints/lstm-with-attention-best-val-512" if "attention" in args else f"./checkpoints/lstm-best-val-512"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_name,
    save_weights_only=True,
    monitor="val_binary_accuracy",
    mode="max",
    save_best_only=True,
)

if TRAIN:
    model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32,
              callbacks=[model_checkpoint_callback], verbose=2)

# Final evaluation
print("Final metrics")
model.load_weights(checkpoint_name)
model.evaluate(X_test, y_test, verbose=2)
