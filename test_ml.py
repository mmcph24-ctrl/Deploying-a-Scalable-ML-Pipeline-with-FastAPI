import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import apply_label
from ml.model import compute_model_metrics, train_model


def test_apply_labels():
    """Test that binary predictions are converted to salary labels."""
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"


def test_train_model():
    """Test that the training function returns a Random Forest model."""
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """Test that model metrics return the expected values."""
    y = np.array([0, 1, 1, 1])
    preds = np.array([0, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)
