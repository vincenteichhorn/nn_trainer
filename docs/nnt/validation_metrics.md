Certainly! Here’s a comprehensive **technical documentation** draft for your metrics module, formatted as a markdown file suitable for a GitHub repo or internal docs.

---

# Metrics Module Documentation

## Overview

This module provides a framework for computing validation metrics in machine learning tasks, focusing on:

* **Classification metrics** based on confusion matrices.
* **Text generation metrics** based on standard NLP metrics like BLEU, ROUGE, METEOR, and NIST.

Metrics are designed to be modular, extensible, and compatible with batch-based predictions encapsulated in the `PredictedBatch` interface.


## ValidationMetric Base Class

`ValidationMetric` is an abstract base class defining the interface for all validation metrics.

### Interface

```python
class ValidationMetric:
    def compute(self, predicted_batch: PredictedBatch) -> None:
        """
        Update the metric state using the predictions and ground truths from predicted_batch.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize and return the computed metrics, typically averaging over all computed batches.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
```

---

## Classification Metrics

### ClassificationMetrics (Abstract Base)

`ClassificationMetrics` computes classification metrics by maintaining a confusion matrix.

#### Key Features

* Maintains a confusion matrix of shape `(num_classes, num_classes)`.
* Computes:

  * Accuracy
  * Precision (per class)
  * Recall (per class)
  * F1 Score (per class)
  * Matthews Correlation Coefficient (MCC) for binary classification

#### Usage

Subclasses must implement:

```python
def check_classification(self, predicted_batch: PredictedBatch) -> List[Tuple[int, int]]:
    """
    Extract (true_label, predicted_label) pairs from a batch.
    """
    raise NotImplementedError
```

The base class `compute` method updates the confusion matrix from these pairs.

---

### OneHotClassificationMetrics

Supports classification tasks where targets are one-hot encoded. Handles:

* **Flat classification:** Direct argmax on logits and labels.
* **Sequence classification:** Supports offset indexing into sequences and ignores padded tokens.
* **Class filtering:** Restricts evaluation to a subset of classes.

#### Initialization Parameters

| Parameter             | Type                      | Description                                      |
| --------------------- | ------------------------- | ------------------------------------------------ |
| `num_classes`         | `int`                     | Number of classes                                |
| `logits_key`          | `str`, default `"logits"` | Key to access logits in prediction               |
| `targets_key`         | `str`, default `"y"`      | Key to access true labels in batch               |
| `sequence_offset`     | `int`, default `0`        | Offset for sequence classification tasks         |
| `classes`             | `List[int]` or `"all"`    | Classes to evaluate (default: all classes)       |
| `label_padding_value` | `int`, default `-100`     | Padding token value for labels in sequence tasks |

---

## Text Generation Metrics

Metrics rely on Hugging Face’s [`evaluate`](https://huggingface.co/docs/evaluate/index) library.

### BleuScore

Computes the BLEU score for generated sequences.

* **`target_key`**: Key in `reference_data` holding ground-truth texts (default `"references"`).
* Accumulates batch scores internally and returns average on `finalize()`.

---

### NistScore

Computes the NIST score, an alternative to BLEU emphasizing informative n-grams.

* Same interface as `BleuScore`.
* Uses `"nist_mt"` metric from `evaluate`.

---

### RougeScore

Computes ROUGE scores including ROUGE-1, ROUGE-2, and ROUGE-L.

* **`target_key`** parameter specifies reference texts.
* Returns averages per ROUGE variant.

---

### MeteorScore

Computes METEOR score, which uses synonym and paraphrase matching.

* Uses the `meteor` metric from `evaluate`.
* Same interface as above metrics.

---

## PredictedBatch Interface

Metrics consume a `PredictedBatch` object which encapsulates:

| Attribute        | Description                                  |
| ---------------- | -------------------------------------------- |
| `prediction`     | Model output (logits, generated texts, etc.) |
| `batch`          | Original input batch (contains ground truth) |
| `reference_data` | Ground-truth data for generation tasks       |

**Example structure for classification:**

```python
PredictedBatch(
    prediction=logits,          # np.ndarray or torch tensor
    batch={"y": labels},        # dict with true labels
    reference_data=None         # reference data from the original provided validation set
)
```

**Example for generation:**

```python
PredictedBatch(
    prediction=["generated sentence 1", ...],
    batch=None,
    reference_data={"references": [["reference sentence 1"], ...]}
)
```

---

## Usage Examples

### Compute Classification Metrics

```python
metrics = OneHotClassificationMetrics(num_classes=5, classes=[0,1,2])

for batch in dataloader:
    pred_batch = PredictedBatch(prediction=logits, batch=batch, reference_data=None)
    metrics.compute(pred_batch)

results = metrics.finalize()
print(results)  # dict with accuracy, precision, recall, f1_score, mcc
```

### Compute BLEU for Generation

```python
bleu = BleuScore()

for batch in generation_batches:
    pred_batch = PredictedBatch(prediction=generated_texts, batch=None, reference_data=batch)
    bleu.compute(pred_batch)

final_bleu = bleu.finalize()
print("BLEU score:", final_bleu["bleu"])
```

---

## Extending the Metrics

To add a new metric:

1. Subclass `ValidationMetric`.
2. Implement:

   * `compute(self, predicted_batch: PredictedBatch)`
   * `finalize(self) -> Dict[str, Any]`
3. Optionally add:

   * `reset(self)` to clear accumulated state.
   * Custom initialization parameters.