import numpy as np
from itertools import groupby


def get_labels_index(labels, symbol):
    return labels.index(symbol)


def label_to_indexes(labels, label):
    res = map(lambda c: get_labels_index(labels, c), label)
    return list(res)


def decode(labels, label):
    binary_label = [a for a, b in groupby(np.argmax(label[0, 2:], 1))]
    return ''.join(labels[c] for c in binary_label if c < len(labels))
