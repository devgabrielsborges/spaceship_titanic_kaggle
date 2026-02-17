"""Linear Regression adapter."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.domain.entities.experiment_config import TaskType
from src.domain.ports.model_port import ModelPort


class LinearRegressionAdapter(ModelPort):
    """Adapter for sklearn LinearRegression."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION):
        self._model: LinearRegression | LogisticRegression | None = None
        self._task_type = task_type
        self._is_classifier = task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )

    @property
    def name(self) -> str:
        return "LogisticRegression" if self._is_classifier else "LinearRegression"

    def build(self, hyperparameters: dict) -> None:
        if self._is_classifier:
            # LogisticRegression has different hyperparameters
            if "positive" in hyperparameters:
                # positive doesn't apply to LogisticRegression
                del hyperparameters["positive"]
            hyperparameters["max_iter"] = hyperparameters.get("max_iter", 1000)
            self._model = LogisticRegression(**hyperparameters)
        else:
            self._model = LinearRegression(**hyperparameters)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        self._model.fit(X, y)

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        return self._model.predict(X)

    def get_search_space(self, trial: Any) -> dict:
        if self._is_classifier:
            # LogisticRegression hyperparameters
            return {
                "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                "solver": trial.suggest_categorical(
                    "solver", ["lbfgs", "liblinear", "saga"]
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 5000),
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
            }
        else:
            # LinearRegression hyperparameters
            return {
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept",
                    [True, False],
                ),
                "positive": trial.suggest_categorical(
                    "positive",
                    [True, False],
                ),
            }

    def get_model(self) -> LinearRegression | LogisticRegression:
        return self._model

    def get_default_trials(self) -> int:
        """
        Linear: 2×2=4 combinations for regression.
        Logistic: more for classification.
        """
        return 30 if self._is_classifier else 10
