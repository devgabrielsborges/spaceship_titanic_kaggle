"""Sklearn-based preprocessor adapter.

Customize this file for each competition's specific feature engineering needs.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.domain.entities.dataset import Dataset
from src.domain.ports.preprocessor_port import PreprocessorPort

SPEND_COLS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]
LUXURY_COLS = ["RoomService", "Spa", "VRDeck"]
BASIC_COLS = ["FoodCourt", "ShoppingMall"]


class SklearnPreprocessor(PreprocessorPort):
    """
    Adapter for sklearn-based preprocessing.

    Customize the transform logic for your specific competition data.
    """

    def __init__(
        self,
        target_col: str | None = None,
        numeric_strategy: str = "knn",
        categorical_strategy: str = "most_frequent",
    ):
        self._scaler = StandardScaler()
        self._ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._knn_imputer = KNNImputer(n_neighbors=5)
        self._ohe_cols = ["HomePlanet", "Destination", "CabinDeck"]
        self._fitted = False
        self._target_col = target_col
        self._numeric_strategy = numeric_strategy
        self._categorical_strategy = categorical_strategy

    # ── Feature engineering ─────────────────────────────────

    @staticmethod
    def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
        """All feature engineering in one place."""
        data = data.copy()

        # -- Cabin features --
        if "Cabin" in data.columns:
            parts = data["Cabin"].str.split("/", expand=True)
            data["CabinDeck"] = parts[0]
            data["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
            data["CabinSide"] = parts[2].map({"P": 1, "S": 0})
            data = data.drop(columns=["Cabin"])

        # -- PassengerId features --
        if "PassengerId" in data.columns:
            group = data["PassengerId"].str.split("_").str[0]
            data["Group"] = pd.to_numeric(group, errors="coerce")
            data["GroupSize"] = data.groupby("Group")["Group"].transform("count")
            data["IsAlone"] = (data["GroupSize"] == 1).astype(int)
            data = data.drop(columns=["Group"])

        # -- Spending features --
        available_spend = [c for c in SPEND_COLS if c in data.columns]
        if available_spend:
            data["TotalSpend"] = data[available_spend].sum(axis=1)
            data["NoSpend"] = (data["TotalSpend"] == 0).astype(int)

        available_luxury = [c for c in LUXURY_COLS if c in data.columns]
        if available_luxury:
            data["LuxurySpend"] = data[available_luxury].sum(axis=1)
            data["LogLuxurySpend"] = np.log1p(data["LuxurySpend"])

        available_basic = [c for c in BASIC_COLS if c in data.columns]
        if available_basic:
            data["BasicSpend"] = data[available_basic].sum(axis=1)
            data["LogBasicSpend"] = np.log1p(data["BasicSpend"])

        # -- Age features --
        if "Age" in data.columns:
            data["IsChild"] = (data["Age"] <= 12).astype(int)
            data["AgeBin"] = pd.cut(
                data["Age"],
                bins=[0, 12, 18, 25, 35, 50, 65, 80],
                labels=False,
            )

        # -- Boolean columns to int --
        if "CryoSleep" in data.columns:
            data["CryoSleep"] = data["CryoSleep"].astype("Int64")
        if "VIP" in data.columns:
            data["VIP"] = data["VIP"].astype("Int64")

        return data

    # ── Imputation ──────────────────────────────────────────

    def _impute_missing_values(
        self, data: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """Impute missing values using KNN imputer for numeric columns."""
        data = data.copy()

        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

        if numeric_cols:
            if fit:
                data[numeric_cols] = self._knn_imputer.fit_transform(data[numeric_cols])
            else:
                data[numeric_cols] = self._knn_imputer.transform(data[numeric_cols])

        # Fill categorical columns with mode
        cat_cols = data.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            if data[col].isna().any():
                mode = data[col].mode()
                fill = mode[0] if len(mode) > 0 else "Unknown"
                data[col] = data[col].fillna(fill)

        return data

    # ── Core pipeline (fit / transform) ─────────────────────

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the preprocessor on the data."""
        data = self._engineer_features(data)
        data = self._impute_missing_values(data, fit=True)

        # Fit and apply OHE so scaler sees final columns
        available_ohe = [c for c in self._ohe_cols if c in data.columns]
        if available_ohe:
            self._ohe.fit(data[available_ohe])
            encoded = self._ohe.transform(data[available_ohe])
            names = self._ohe.get_feature_names_out(available_ohe)
            ohe_df = pd.DataFrame(encoded, columns=names, index=data.index)
            data = pd.concat([data.drop(columns=available_ohe), ohe_df], axis=1)

        numeric_cols = data.select_dtypes(include=["number"]).columns
        self._scaler.fit(data[numeric_cols])
        self._fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted preprocessor."""
        data = self._engineer_features(data)
        data = self._impute_missing_values(data, fit=False)

        available_ohe = [c for c in self._ohe_cols if c in data.columns]
        if available_ohe:
            encoded = self._ohe.transform(data[available_ohe])
            names = self._ohe.get_feature_names_out(available_ohe)
            ohe_df = pd.DataFrame(encoded, columns=names, index=data.index)
            data = pd.concat([data.drop(columns=available_ohe), ohe_df], axis=1)

        numeric_cols = data.select_dtypes(include=["number"]).columns
        data[numeric_cols] = self._scaler.transform(data[numeric_cols])
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    # ── Train/test split ────────────────────────────────────

    def get_train_test_split(
        self,
        data: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        id_col: str | None = "id",
    ) -> Dataset:
        """
        Preprocess, split, and scale data into a Dataset entity.

        Args:
            data: Raw dataframe.
            target_col: Name of the target column.
            test_size: Fraction of data for testing.
            random_state: Random seed.
            id_col: ID column to exclude from features (None to skip).

        Returns:
            Dataset entity with train/test splits.
        """
        data = data.copy()

        # Feature engineering
        data = self._engineer_features(data)

        # Impute with fresh imputer
        self._knn_imputer = KNNImputer(n_neighbors=5)
        data = self._impute_missing_values(data, fit=True)

        # One-hot encode
        available_ohe = [c for c in self._ohe_cols if c in data.columns]
        if available_ohe:
            self._ohe.fit(data[available_ohe])
            encoded = self._ohe.transform(data[available_ohe])
            names = self._ohe.get_feature_names_out(available_ohe)
            ohe_df = pd.DataFrame(encoded, columns=names, index=data.index)
            data = pd.concat(
                [data.drop(columns=available_ohe), ohe_df],
                axis=1,
            )

        # Drop non-feature columns
        drop_cols = [target_col, "PassengerId", "Name"]
        if id_col and id_col in data.columns:
            drop_cols.append(id_col)
        drop_cols = [c for c in drop_cols if c in data.columns]

        X = data.drop(columns=drop_cols)
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale numeric features (fit on train only)
        numeric_cols = X_train.select_dtypes(include=["number"]).columns
        self._scaler.fit(X_train[numeric_cols])
        X_train[numeric_cols] = self._scaler.transform(X_train[numeric_cols])
        X_test[numeric_cols] = self._scaler.transform(X_test[numeric_cols])

        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
