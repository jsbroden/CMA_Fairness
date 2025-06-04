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


# Anlayze CP sets


def summarize_for_predicate(cp_df, predicate, description):
    """
    Filters cp_df by rows where pred_set satisfies `predicate`,
    then computes proportions of various flags among that subset.

    predicate: a function that takes one set s and returns True/False.
    description: string to print in the header.
    """
    subset = cp_df[cp_df["pred_set"].apply(predicate)]
    n_subset = len(subset)
    if n_subset == 0:
        print(f"No cases where pred_set {description}.")
        return

    prop_true_label_1 = (subset["true_label"] == 1).sum() / n_subset
    prop_frau1_1 = (subset["frau1"] == 1).sum() / n_subset
    prop_ng_1 = (subset["nongerman"] == 1).sum() / n_subset
    prop_ng_male_1 = (subset["nongerman_male"] == 1).sum() / n_subset
    prop_ng_female_1 = (subset["nongerman_female"] == 1).sum() / n_subset

    print(f"Among cases where pred_set {description}:")
    print(f"  Proportion true_label == 1:        {prop_true_label_1:.3f}")
    print(f"  Proportion frau1 == 1:             {prop_frau1_1:.3f}")
    print(f"  Proportion nongerman == 1:         {prop_ng_1:.3f}")
    print(f"  Proportion nongerman_male == 1:    {prop_ng_male_1:.3f}")
    print(f"  Proportion nongerman_female == 1:  {prop_ng_female_1:.3f}")
    print()


def summarize_by_indicator(df, indicator_col, positive_label, negative_label):
    """
    Groups df by indicator_col (0 vs 1), sums the is_ambiguous/is_zero_only/
    is_one_only flags,then returns both raw counts and percentages
    (out of each subgroup's size).

    positive_label/negative_label: names to assign to index 1 and 0
    respectively.
    """
    df = df.copy()
    df["is_ambiguous"] = df["pred_set"].apply(lambda s: s == {0, 1})
    df["is_zero_only"] = df["pred_set"].apply(lambda s: s == {0})
    df["is_one_only"] = df["pred_set"].apply(lambda s: s == {1})

    sub = df.dropna(subset=[indicator_col])

    # Compute raw counts
    counts = (
        sub.groupby(indicator_col)[["is_ambiguous", "is_zero_only", "is_one_only"]]
        .sum()
        .rename(index={0: negative_label, 1: positive_label})
    )

    # Compute subgroup sizes (number of rows where indicator == 0 or 1)
    sizes = (
        sub[indicator_col]
        .value_counts()
        .rename(index={0: negative_label, 1: positive_label})
    )
    percentages = counts.div(sizes, axis=0) * 100

    return counts, percentages
