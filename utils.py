import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

# Metrics


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


# CP


def compute_nc_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    For each calibration example, return 1 - p_true.
    probs: array of shape (n_samples, 2)
    labels: array of shape (n_samples,) with values in {0, 1}
    """
    p_true = probs[np.arange(len(labels)), labels]
    return 1.0 - p_true


# ??? Alternative scoring function negative log-probabilites??


def find_threshold(nonconformity: np.ndarray, alpha: float) -> float:
    """
    Return the (1-alpha)-quantile (higher interpolation) of the nc scores.
    """
    return np.quantile(nonconformity, 1 - alpha, interpolation="higher")


def predict_conformal_sets(model, X: pd.DataFrame, q_hat: float) -> np.ndarray:
    """
    For each row in X, compute the conformal prediction set.
    Returns a list of sets.
    """
    probs = model.predict_proba(X)  # shape (n, 2)
    nonconf_matrix = 1.0 - probs  # shape (n, 2): nc score for label = 0, 1
    # include c whenver nonconf_matrix[i,c] <= q_hat
    return [set(np.where(nc_row <= q_hat)[0]) for nc_row in nonconf_matrix]


def evaluate_sets(pred_sets: list, y_true: pd.Series) -> dict:
    """
    Compute empirical coverage and average set size.
    """
    hits = [y_true.iloc[i] in pred_sets[i] for i in range(len(y_true))]
    coverage = np.mean(hits)
    avg_size = np.mean([len(s) for s in pred_sets])
    return {"coverage": coverage, "avg_size": avg_size}


#


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
