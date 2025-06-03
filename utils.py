import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def precision_at_k(y_true, y_score, k):
    """
    Among all instances that were marked 1 (the top k fraction by score),
    what fraction actually had y_true = 1
    prec = TP/(TP + FP)
    """
    threshold = np.sort(y_score)[::-1][int(k * len(y_score))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_score])
    return precision_score(y_true, y_pred)


def recall_at_k(y_true, y_score, k):
    """
    Among all actual positives in y_true (where y_true = 1), what fraction
    did our “top k” rule correctly include as predicted positives?
    rec = TP/(TP + FN)
    """
    threshold = np.sort(y_score)[::-1][int(k * len(y_score))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_score])
    return recall_score(y_true, y_pred)


def summarize_by_predset(cp_df):
    """
    Given a DataFrame cp_df with columns:
      - 'pred_set'        (each cell is a Python set {0}, {1}, or {0,1})
      - 'true_label'      (0/1)
      - 'frau1'           (0/1)
      - 'nongerman'       (0/1 or NaN)
      - 'nongerman_male'  (0/1)
      - 'nongerman_female'(0/1)
    This prints (or returns) for each of the three categories:
      {0}, {1}, and {0,1}:
        • proportion where true_label==1
        • proportion where frau1==1
        • proportion where nongerman==1
        • proportion where nongerman_male==1
        • proportion where nongerman_female==1
    """

    def _subset_props(df, desc):
        n = len(df)
        if n == 0:
            return {"desc": desc, "n": 0}

        return {
            "desc": desc,
            "n": n,
            "p_true_1": (df["true_label"] == 1).mean(),
            "p_frau1_1": (df["frau1"] == 1).mean(),
            "p_nongerman_1": (df["nongerman"] == 1).mean(),
            "p_nongerman_male_1": (df["nongerman_male"] == 1).mean(),
            "p_nongerman_female_1": (df["nongerman_female"] == 1).mean(),
        }

    results = []
    for s, label in [({0}, "{0}"), ({1}, "{1}"), ({0, 1}, "{0,1}")]:
        sub = cp_df[cp_df["pred_set"].apply(lambda x: set(x) == s)]
        results.append(_subset_props(sub, f"pred_set == {label}"))

    return pd.DataFrame(results)
