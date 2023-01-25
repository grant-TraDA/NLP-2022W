import numpy as np
from os import path



def get_bin(length, bins=[0, 50, 75, 100, 500]):
  bins = np.array(bins)
  bin = np.argmin(length >= bins) - 1
  if bin == -1: 
    bin = len(bins) -1
  return bin


def get_length_sentence_row(sentence, type):
  bin = get_bin(len(sentence))
  return f'{type}	{bin}	{sentence}'


def create_sentence_length_probing_file(path, train_samples, test_samples):
    rows = []
    for s in train_samples:
        rows.append(get_length_sentence_row(s.texts[0], 'tr'))
        rows.append(get_length_sentence_row(s.texts[1], 'tr'))

    k = len(test_samples) // 2
    for s in test_samples[0:k]:
        rows.append(get_length_sentence_row(s.texts[0], 'va'))
        rows.append(get_length_sentence_row(s.texts[1], 'va'))

    for s in test_samples[k:]:
        rows.append(get_length_sentence_row(s.texts[0], 'te'))
        rows.append(get_length_sentence_row(s.texts[1], 'te'))

    with open(path.join(path, "sentence_lenght.txt"), "a") as f:
        f.write('\n'.join(rows))



  