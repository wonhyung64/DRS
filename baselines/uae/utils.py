import numpy as np


def binarize(y, thres=3):
    """Given threshold, binarize the ratings.
    """
    y[y< thres] = 0
    y[y>=thres] = 1

    return y
