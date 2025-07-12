import pytest
import numpy as np
from nnt.validation_metrics.classification_metrics import ClassificationMetrics, OneHotClassificationMetrics

# File: tests/validation_metrics/test_validation_metrics.py


class DummyPredictedBatch:
    def __init__(self, logits, labels, logits_key="logits", targets_key="y"):
        class DummyPrediction:
            pass

        self.prediction = DummyPrediction()
        setattr(self.prediction, logits_key, logits)
        self.batch = {targets_key: labels}


class DummyClassificationMetrics(ClassificationMetrics):
    def check_classification(self, predicted_batch):
        # Just return zipped argmax for testing
        logits = vars(predicted_batch.prediction).get("logits")
        labels = predicted_batch.batch.get("y")
        true_labels = np.argmax(labels, axis=1)
        pred_labels = np.argmax(logits, axis=1)
        return list(zip(true_labels, pred_labels))


def test_classificationmetrics_init():
    metric = DummyClassificationMetrics(num_classes=3)
    assert metric.num_classes == 3
    assert metric.confusion_matrix.shape == (3, 3)


def test_classificationmetrics_compute_and_finalize():
    metric = DummyClassificationMetrics(num_classes=2)
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([[0, 1], [1, 0]])
    batch = DummyPredictedBatch(logits, labels)
    metric.compute(batch)
    # Should update confusion matrix
    assert np.sum(metric.confusion_matrix) == 2
    result = metric.finalize()
    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1_score" in result
    assert "mcc" in result


def test_classificationmetrics_abstract_check():
    metric = ClassificationMetrics(num_classes=2)
    with pytest.raises(NotImplementedError):
        metric.check_classification(None)


def test_onehotclassificationmetrics_init():
    metric = OneHotClassificationMetrics(num_classes=4, classes=[1, 2])
    assert metric.num_classes == 4
    assert metric.classes == [1, 2]
    assert metric.sequence_offset == 0


def test_onehotclassificationmetrics_check_classification_2d():
    metric = OneHotClassificationMetrics(num_classes=3)
    logits = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
    labels = np.array([[0, 1, 0], [1, 0, 0]])
    batch = DummyPredictedBatch(logits, labels)
    pairs = metric.check_classification(batch)
    assert pairs == [(1, 1), (0, 1)]


def test_onehotclassificationmetrics_check_classification_3d():
    metric = OneHotClassificationMetrics(num_classes=3, sequence_offset=1)
    logits = np.array([[[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]], [[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]]])
    labels = np.array([[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]])
    batch = DummyPredictedBatch(logits, labels)
    pairs = metric.check_classification(batch)
    assert pairs == [(0, 1), (1, 0)]


def test_onehotclassificationmetrics_check_classification_invalid_shape():
    metric = OneHotClassificationMetrics(num_classes=3)
    logits = np.ones((2,))  # Invalid shape
    labels = np.ones((2, 3))
    batch = DummyPredictedBatch(logits, labels)
    with pytest.raises(ValueError):
        metric.check_classification(batch)


def test_onehotclassificationmetrics_compute_and_finalize():
    metric = OneHotClassificationMetrics(num_classes=2)
    logits = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([[1, 0], [0, 1]])
    batch = DummyPredictedBatch(logits, labels)
    metric.compute(batch)
    result = metric.finalize()
    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1_score" in result
    assert "mcc" in result
