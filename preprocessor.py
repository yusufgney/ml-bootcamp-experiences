import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HousePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):

        num_cols = X.select_dtypes(["int64", "float64"]).columns
        cat_cols = X.select_dtypes(["object"]).columns

        self.num_medians_ = {  # sona '_' koymak -> fit sirasinda ogrenildi anlami tasir.
            col: X[col].median() for col in num_cols
        }

        self.cat_mode_ = {
            col: X[col].mode()[0] for col in cat_cols
        }

        self.top_locations_ = X["location"].value_counts().nlargest(20).index

        return self

    def transform(self, X):
        X = X.copy()

        for col, val in self.num_medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        for col, val in self.cat_mode_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        if "furnishing" in X.columns:
            X["furnishing"] = X["furnishing"].map({"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2})

        if "location" in X.columns:
            X["location"] = X["location"].apply(lambda x: x if x in self.top_locations_ else "other")

        return X