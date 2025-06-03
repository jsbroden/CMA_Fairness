import numpy as np
from sklearn.metrics import precision_score, recall_score

"""
Among all instances that were marked 1 (the top k fraction by score),
what fraction actually had y_true = 1

prec = TP/(TP + FP)
"""


def precision_at_k(y_true, y_score, k):
    threshold = np.sort(y_score)[::-1][int(k * len(y_score))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_score])
    return precision_score(y_true, y_pred)


"""
Among all actual positives in y_true (where y_true = 1),
what fraction did our “top k” rule correctly include as predicted positives?

rec = TP/(TP + FN)
"""


def recall_at_k(y_true, y_score, k):
    threshold = np.sort(y_score)[::-1][int(k * len(y_score))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_score])
    return recall_score(y_true, y_pred)
