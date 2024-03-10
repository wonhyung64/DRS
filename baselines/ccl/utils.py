import numpy as np


def binarize(y, thres=3):
    """Given threshold, binarize the ratings.
    """
    y[y< thres] = 0
    y[y>=thres] = 1

    return y


def shuffle(x, y):
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)

    return x[idxs], y[idxs]
