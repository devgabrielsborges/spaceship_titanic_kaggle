"""SVM adapter."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC

from src.domain.entities.experiment_config import TaskType
from src.domain.ports.model_port import ModelPort


class SVMAdapter(ModelPort):
    """Adapter for sklearn Support Vector Machine."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION):
        self._model: SVR | SVC | None = None
        self._task_type = task_type
        self._is_classifier = task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )

    @property
    def name(self) -> str:
        return "SVM"

    def build(self, hyperparameters: dict) -> None:
        if self._is_classifier:
            # SVC uses different parameter names
            if "epsilon" in hyperparameters:
                del hyperparameters["epsilon"]  # epsilon doesn't apply to SVC
            self._model = SVC(**hyperparameters, probability=True)
        else:
            self._model = SVR(**hyperparameters)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_search_space(self, trial: Any) -> dict:
        base_params = {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical(
                "kernel",
                ["linear", "rbf", "poly"],
            ),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

        # Only add epsilon for regression
        if not self._is_classifier:
            base_params["epsilon"] = trial.suggest_float("epsilon", 0.01, 1.0)

        return base_params

    def get_model(self) -> SVR | SVC:
        return self._model

    def get_default_trials(self) -> int:
        """SVM has medium complexity: C × 3 kernels × 2 gamma × epsilon."""
        return 50
