import pandas as pd
import numpy as np
from fanova import fANOVA
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class MultiverseFanova:
    def __init__(self, X_choices: pd.DataFrame, y_values: np.ndarray):
        """
        X_choices: DataFrame with one column per analytical choice (categorical).
        y_values: 1D array of outcome values (e.g., ambiguity).
        """
        self.X = X_choices.copy()
        self.y = y_values

        # Build configuration space from categorical choices
        self.cs = ConfigurationSpace()
        for col in self.X.columns:
            categories = sorted(self.X[col].unique())
            self.cs.add_hyperparameter(CategoricalHyperparameter(col, categories))

        # Encode categorical choices numerically
        self.X_encoded = (
            self.X.astype("category").apply(lambda col: col.cat.codes).to_numpy()
        )
        self.encoder = {
            col: dict(enumerate(self.X[col].astype("category").cat.categories))
            for col in self.X.columns
        }

        # Fit FANOVA model
        self.fanova = fANOVA(self.X_encoded, self.y, config_space=self.cs)

    def quantify_individual_importance(self):
        """
        Returns a DataFrame with the individual variance explained by each
        analytical decision.
        """
        importances = {}
        for i, col in enumerate(self.X.columns):
            result = self.fanova.quantify_importance((i,))
            importances[col] = result

        # Format into DataFrame
        df = pd.DataFrame.from_dict(importances, orient="index")
        df.index.name = "decision"
        return df


# Example usage:
# fanova = MultiverseFanova(
# X_choices=df[["model", "feature_set"]], y_values=df["ambiguity"])
# fanova.quantify_individual_importance()
