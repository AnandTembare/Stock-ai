from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    model: Any
    model_name: str
    train_index: pd.Index
    test_index: pd.Index
    probabilities: pd.Series
    predictions: pd.Series
    metrics: dict[str, float | int | None]
    roc_curve: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_columns: list[str]


def train_time_series_classifier(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ModelResult:
    data = feature_frame.dropna(subset=feature_columns + ["Target"]).copy()
    if len(data) < 120:
        raise ValueError("At least 120 feature rows are required for a stable train/test split.")

    n_test = int(len(data) * test_size)
    n_test = min(max(n_test, 40), len(data) - 80)
    split_at = len(data) - n_test

    train = data.iloc[:split_at]
    test = data.iloc[split_at:]
    x_train = train[feature_columns]
    y_train = train["Target"].astype(int)
    x_test = test[feature_columns]
    y_test = test["Target"].astype(int)

    model, fitted_name = build_model(model_name, random_state=random_state)
    model.fit(x_train, y_train)
    probabilities = _positive_class_probability(model, x_test)
    predictions = (probabilities >= 0.5).astype(int)

    probability_series = pd.Series(probabilities, index=test.index, name="Buy_Probability")
    prediction_series = pd.Series(predictions, index=test.index, name="Prediction")
    metrics = classification_metrics(y_test.to_numpy(), probabilities, predictions)
    roc = roc_curve_frame(y_test.to_numpy(), probabilities)
    matrix = confusion_matrix_frame(y_test.to_numpy(), predictions)

    return ModelResult(
        model=model,
        model_name=fitted_name,
        train_index=train.index,
        test_index=test.index,
        probabilities=probability_series,
        predictions=prediction_series,
        metrics=metrics,
        roc_curve=roc,
        confusion_matrix=matrix,
        feature_columns=feature_columns,
    )


def build_model(model_name: str, random_state: int = 42) -> tuple[Any, str]:
    try:
        return _build_sklearn_model(model_name, random_state), model_name
    except ImportError:
        return NumpyLogisticClassifier(learning_rate=0.08, epochs=900), "Logistic Regression (numpy fallback)"


def _build_sklearn_model(model_name: str, random_state: int) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state),
            ),
        ]
    )
    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=7,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    gradient = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=180,
                    learning_rate=0.045,
                    max_depth=3,
                    random_state=random_state,
                ),
            ),
        ]
    )

    options = {
        "Logistic Regression": logistic,
        "Random Forest": forest,
        "Gradient Boosting": gradient,
        "Soft Voting Ensemble": VotingClassifier(
            estimators=[("lr", logistic), ("rf", forest), ("gb", gradient)],
            voting="soft",
            weights=[1, 2, 2],
        ),
    }
    return options.get(model_name, options["Soft Voting Ensemble"])


def latest_probability(model: Any, latest_features: pd.DataFrame) -> float:
    return float(_positive_class_probability(model, latest_features)[0])


def _positive_class_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, 1]
    decision = model.decision_function(x)
    return 1 / (1 + np.exp(-decision))


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float | int | None]:
    y_true = y_true.astype(int)
    predictions = predictions.astype(int)
    tp = int(((y_true == 1) & (predictions == 1)).sum())
    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())
    total = max(len(y_true), 1)

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = binary_auc(y_true, probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_curve_frame(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame({"False Positive Rate": [0, 1], "True Positive Rate": [0, 1]})

    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    rows = []
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = ((y_true == 1) & predicted).sum()
        fp = ((y_true == 0) & predicted).sum()
        rows.append(
            {
                "Threshold": threshold,
                "False Positive Rate": fp / negatives,
                "True Positive Rate": tp / positives,
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_frame(y_true: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())
    tp = int(((y_true == 1) & (predictions == 1)).sum())
    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Down", "Actual Up"],
        columns=["Pred Down", "Pred Up"],
    )


def feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    final_model = model
    if hasattr(model, "named_steps"):
        final_model = model.named_steps.get("model", model)

    values = None
    if hasattr(final_model, "feature_importances_"):
        values = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        values = np.abs(final_model.coef_).ravel()

    if values is None or len(values) != len(feature_columns):
        return pd.DataFrame(columns=["Feature", "Importance"])

    frame = pd.DataFrame({"Feature": feature_columns, "Importance": values})
    return frame.sort_values("Importance", ascending=False).head(12)


class NumpyLogisticClassifier:
    """Small fallback classifier used only when scikit-learn is unavailable."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 700) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "NumpyLogisticClassifier":
        values = x.to_numpy(dtype=float)
        target = y.to_numpy(dtype=float)
        self.mean_ = np.nanmean(values, axis=0)
        self.std_ = np.nanstd(values, axis=0)
        self.std_[self.std_ == 0] = 1
        values = np.nan_to_num((values - self.mean_) / self.std_)
        design = np.c_[np.ones(len(values)), values]
        weights = np.zeros(design.shape[1])

        for _ in range(self.epochs):
            probabilities = 1 / (1 + np.exp(-design @ weights))
            gradient = design.T @ (probabilities - target) / len(target)
            weights -= self.learning_rate * gradient

        self.weights_ = weights
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        values = x.to_numpy(dtype=float)
        values = np.nan_to_num((values - self.mean_) / self.std_)
        design = np.c_[np.ones(len(values)), values]
        probabilities = 1 / (1 + np.exp(-design @ self.weights_))
        return np.c_[1 - probabilities, probabilities]

