from hamilton import function_modifiers
from sklearn import base, linear_model, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict


# --------------------------------------------------------------------------------------#
# MODEL INIT
# --------------------------------------------------------------------------------------#
@function_modifiers.config.when(classifier="svm")
def model__svm(gamma: float = 0.001) -> base.ClassifierMixin:
    return svm.SVC(gamma=gamma)


@function_modifiers.config.when(classifier="logistic")
def model__logistic_regression(penalty: str) -> base.ClassifierMixin:
    return linear_model.LogisticRegression(penalty)


# --------------------------------------------------------------------------------------#
# SPLIT AND PREPROCESS
# --------------------------------------------------------------------------------------#
@function_modifiers.extract_fields(
    {
        "X_train": np.ndarray,
        "X_test": np.ndarray,
        "y_train": np.ndarray,
        "y_test": np.ndarray,
    }
)
def train_test_split_func(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    test_size_fraction: float,
    shuffle_train_test_split: bool,
) -> Dict[str, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix,
        target,
        test_size=test_size_fraction,
        shuffle=shuffle_train_test_split,
        random_state=42,
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def y_test_with_labels(y_test: np.ndarray, target_names: np.ndarray) -> np.ndarray:
    return np.array([target_names[idx] for idx in y_test])


# --------------------------------------------------------------------------------------#
# TRAIN
# --------------------------------------------------------------------------------------#
def fit_model(
    model: base.ClassifierMixin, X_train: np.ndarray, y_train: np.ndarray
) -> base.ClassifierMixin:
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------------------#
# PREDICT
# --------------------------------------------------------------------------------------#
def predicted_output(fit_model: base.ClassifierMixin, X_test: np.ndarray) -> np.ndarray:
    return fit_model.predict(X_test)


def predicted_output_with_labels(
    predicted_output: np.ndarray, target_names: np.ndarray
) -> np.ndarray:
    return np.array([target_names[idx] for idx in predicted_output])


def classification_report(
    predicted_output_with_labels: np.ndarray, y_test_with_labels: np.ndarray
) -> str:
    return metrics.classification_report(
        y_test_with_labels, predicted_output_with_labels
    )


def confusion_matrix(
    predicted_output_with_labels: np.ndarray, y_test_with_labels: np.ndarray
) -> str:
    return metrics.confusion_matrix(y_test_with_labels, predicted_output_with_labels)
