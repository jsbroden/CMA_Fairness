# File: pipeline_factory.py

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    KBinsDiscretizer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# -----------------------------------------------------------------
PROTECTED_COLS = ["frau1", "maxdeutsch1", "maxdeutsch.Missing."]
SLICE = slice(4, 164)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# 1. Build the ColumnTransformer in one place
# -----------------------------------------------------------------
def build_preprocessor(
    df, universe, slice_cols=SLICE, protected_cols=PROTECTED_COLS
):
    """
    Inspect *df* once, decide which columns are numeric / categorical,
    and assemble a ColumnTransformer that reflects every preprocessing
    choice encoded in *universe*.
    """
    # -- 1A. Select base feature columns (+ optional protected attrs)
    feat_cols = list(df.columns[slice_cols])
    if universe["feature_set"] == "without_protected":
        feat_cols = [c for c in feat_cols if c not in protected_cols]

    # -- 1B. Split by dtype  (object / category → categorical)
    cat_cols = [
        c
        for c in feat_cols
        if (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))
    ]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    # -- 1C. Decide scalar / binning per universe -------------------
    scaler = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "none": "passthrough",
    }.get(universe.get("scaling", "none"))

    binning_choice = universe.get("binning", "none")
    if binning_choice.startswith("kbins_"):
        n_bins = int(binning_choice.split("_")[1])
        binning = KBinsDiscretizer(
            n_bins=n_bins, encode="onehot-dense", strategy="uniform"
        )
    else:
        binning = "passthrough"

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", scaler),
            ("bin", binning),
        ]
    )

    # -- 1D. Encoding decision --------------------------------------
    encoder = {
        "onehot": OneHotEncoder(handle_unknown="ignore"),
        "ordinal": OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        ),
        "none": "passthrough",
    }.get(universe.get("encoding", "onehot"))

    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("encode", encoder)]
    )

    # -- 1E. Assemble column transformer ----------------------------
    pre = ColumnTransformer(
        [("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)],
        remainder="drop",
    )

    return pre, feat_cols


# -----------------------------------------------------------------


# -----------------------------------------------------------------
# 2. Minimal outer wrapper (row‑filter + fit/transform)
# -----------------------------------------------------------------
def make_train_test_matrices(raw_train, raw_test, universe, slice_cols=SLICE):
    """
    Returns X_train, X_test, y_train, y_test, and the fitted preprocessor.
    Row filtering and feature preprocessing are entirely governed by *universe*.
    """
    # ---- A. Row‑level subgroup filter -----------------------------
    if universe["exclude_subgroups"] == "drops-non-german":
        raw_train = raw_train[raw_train["nongerman"] != 1]
        raw_test = raw_test[raw_test["nongerman"] != 1]

    # ---- B. Build & fit the preprocessor --------------------------
    pre, feat_cols = build_preprocessor(raw_train, universe, slice_cols=slice_cols)

    X_train = pre.fit_transform(raw_train[feat_cols])
    X_test = pre.transform(raw_test[feat_cols])

    y_train = raw_train["target"].values
    y_test = raw_test["target"].values
    return X_train, X_test, y_train, y_test, pre


# -----------------------------------------------------------------
