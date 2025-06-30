import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from collections import defaultdict
from aif360.metrics import ClassificationMetric
import matplotlib.pyplot as plt


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
# -> since $-\log p$ is a monotonic transform of $p$, it would
# lead to the same ordering of calibration scores and an
# equivalent conformal threshold in this binary case.)


def find_threshold(nonconformity: np.ndarray, alpha: float) -> float:
    """
    Return the (1-alpha)-quantile (higher interpolation) of the nc scores.
    """
    return np.quantile(nonconformity, 1 - alpha, method="higher")


def predict_conformal_sets(model, X: pd.DataFrame, q_hat: float) -> list[set[int]]:
    """
    For each row in X, compute the conformal prediction set.
    Returns a list of sets.
    """
    probs = model.predict_proba(X)  # shape (n, 2)
    nonconf_matrix = 1.0 - probs  # shape (n, 2): nc score for label = 0, 1
    # include c whenver nonconf_matrix[i,c] <= q_hat
    return [set(np.where(nc_row <= q_hat)[0]) for nc_row in nonconf_matrix]


# add? Ensures non-empty sets by including the top class if needed.


def evaluate_sets(pred_sets: list, y_true: pd.Series) -> dict:
    """
    Compute empirical coverage and average set size.
    """
    hits = [y_true.iloc[i] in pred_sets[i] for i in range(len(y_true))]
    coverage = np.mean(hits)
    avg_size = np.mean([len(s) for s in pred_sets])
    return {"coverage": coverage, "avg_size": avg_size}


# Analyze CP sets


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


# Performance evaluation


def aif_test(dataset, scores, thresh_arr, unpriv_group, priv_group):
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (scores > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            dataset,
            dataset_pred,
            unprivileged_groups=unpriv_group,
            privileged_groups=priv_group,
        )

        metric_arrs["bal_acc"].append(
            (metric.true_positive_rate() + metric.true_negative_rate()) / 2
        )
        metric_arrs["f1"].append(
            2
            * (metric.precision() * metric.recall())
            / (metric.precision() + metric.recall())
        )
        metric_arrs["acc"].append(metric.accuracy())
        metric_arrs["prec"].append(metric.precision())
        metric_arrs["rec"].append(metric.recall())
        metric_arrs["err_rate"].append(metric.error_rate())
        metric_arrs["err_rate_diff"].append(metric.error_rate_difference())
        metric_arrs["disp_imp"].append(metric.disparate_impact())
        metric_arrs["stat_par_diff"].append(metric.statistical_parity_difference())
        metric_arrs["eq_opp_diff"].append(metric.equal_opportunity_difference())
        metric_arrs["avg_odds_diff"].append(metric.average_odds_difference())

    return metric_arrs


def aif_plot(
    x,
    x_name,
    y_left,
    y_left_name,
    y_right,
    y_right_name,
    cutoff1,
    cutoff2,
    ax1min=0,
    ax1max=1,
    ax2min=0,
    ax2max=1,
):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(x, y_left, color="steelblue")
    ax1.set_xlabel(x_name, fontsize=18)
    ax1.set_ylabel(y_left_name, color="steelblue", fontsize=18)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.set_ylim(ax1min, ax1max)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color="r")
    ax2.set_ylabel(y_right_name, color="r", fontsize=18)
    ax2.set_ylim(ax2min, ax2max)

    ax2.axvline(cutoff1, color="k", linestyle="dashed")
    ax2.axvline(cutoff2, color="k", linestyle="dotted")
    ax2.text(cutoff1 - 0.015, ax1max + 0.015, "P1b", fontsize=16)
    ax2.text(cutoff2 - 0.015, ax1max + 0.015, "P1a", fontsize=16)

    ax2.yaxis.set_tick_params(labelsize=16)
    ax2.grid(True)


def aif_plot2(
    x,
    x_name,
    y_left,
    y_left2,
    y_left_name,
    y_right,
    y_right2,
    y_right_name,
    cutoff1,
    cutoff12,
    cutoff2,
    cutoff22,
    ax1min=0,
    ax1max=1,
    ax2min=0,
    ax2max=1,
):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(x, y_left, color="steelblue")
    ax1.plot(x, y_left2, color="steelblue", linestyle="dashdot")
    ax1.set_xlabel(x_name, fontsize=18)
    ax1.set_ylabel(y_left_name, color="steelblue", fontsize=18)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.set_ylim(ax1min, ax1max)

    # fix! understand purpose and uncomment again:
    # ax1.legend(handles=legend_elements, fontsize=16)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color="r")
    ax2.plot(x, y_right2, color="r", linestyle="dashdot")
    ax2.set_ylabel(y_right_name, color="r", fontsize=18)
    ax2.set_ylim(ax2min, ax2max)

    ax2.axvline(cutoff1, color="k", linestyle="dashed")
    ax2.axvline(cutoff2, color="k", linestyle="dotted")
    ax2.text(cutoff1, ax1max + 0.015, "P1b-l", fontsize=16)
    ax2.text(cutoff2, ax1max + 0.015, "P1a-l", fontsize=16)

    ax2.axvline(cutoff12, color="gray", linestyle="dashed")
    ax2.axvline(cutoff22, color="gray", linestyle="dotted")
    ax2.text(
        cutoff12 - 0.055, ax1max + 0.015, "P1b-s", fontsize=16, color="gray"
    )  # cutoff12 - 0.025
    ax2.text(cutoff22 - 0.025, ax1max + 0.015, "P1a-s", fontsize=16, color="gray")

    ax2.yaxis.set_tick_params(labelsize=16)
    ax2.grid(True)


# multiverse, added 30.6.
def apply_universe_filters(df, universe):
    """
    Apply all data‑level decisions encoded in one universe dict.
    Currently supports:
      • exclude_subgroups  – { 'keep-all', 'drop-non-german' }
    Returns a *copy* of df with rows dropped accordingly.
    """
    df = df.copy()

    # ---- (A) row‑level exclusion ------------------------------------------
    if universe["exclude_subgroups"] == "drop-non-german":
        df = df[df["nongerman"] != 1]  # keep rows where nongerman == 0

    # (...other subgroup filters can be added here...)
    return df
