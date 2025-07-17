from nnt.validation_metrics.validation_metric import ValidationMetric
import evaluate
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nnt.validators.validator import PredictedBatch


class BleuScore(ValidationMetric):
    """
    BleuScore is a validation metric for evaluating the quality of text generation tasks using BLEU score.

    Args:
        target_key (str): The key in the predicted batch that contains the target text.

    Methods:
        compute(predicted_batch): Computes the BLEU score for the predicted batch.
        finalize(): Finalizes the BLEU score computation and returns the average score.
    """

    def __init__(self, target_key: str = "references"):
        """
        Initialize the BleuScore metric with the target key and BLEU evaluator.

        Args:
            target_key (str): Key for reference texts in the batch.
        """
        self.bleu = evaluate.load("bleu")
        self.target_key = target_key
        self.scores = []

    def compute(self, predicted_batch: "PredictedBatch") -> None:
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        scores = self.bleu.compute(predictions=generations, references=references)
        self.scores.extend([scores["bleu"]] * len(generations))

    def finalize(self) -> dict:
        """
        Finalize the BLEU score computation and return the average score.

        Returns:
            dict: Dictionary with the average BLEU score.
        """
        if not self.scores:
            return {"bleu": 0.0}
        out = {"bleu": sum(self.scores) / len(self.scores)}
        self.scores = []
        return out


class NistScore(ValidationMetric):
    """
    NistScore is a validation metric for evaluating the quality of text generation tasks using NIST score.

    Args:
        target_key (str): The key in the predicted batch that contains the target text.

    Methods:
        compute(predicted_batch): Computes the NIST score for the predicted batch.
        finalize(): Finalizes the NIST score computation and returns the average score.
    """

    def __init__(self, target_key: str = "references"):
        """
        Initialize the NistScore metric with the target key and NIST evaluator.

        Args:
            target_key (str): Key for reference texts in the batch.
        """
        self.nist = evaluate.load("nist_mt")
        self.target_key = target_key
        self.scores = []

    def compute(self, predicted_batch: "PredictedBatch") -> None:
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        try:
            scores = self.nist.compute(predictions=generations, references=references)
        except Exception as e:
            print(f"Error computing NIST score: {e}")
            scores = {"nist_mt": 0.0}
        self.scores.extend([scores["nist_mt"]] * len(generations))

    def finalize(self) -> dict:
        """
        Finalize the NIST score computation and return the average score.

        Returns:
            dict: Dictionary with the average NIST score.
        """
        if not self.scores:
            return {"nist": 0.0}
        out = {"nist": sum(self.scores) / len(self.scores)}
        self.scores = []
        return out


class RougeScore(ValidationMetric):
    """
    RougeScore is a validation metric for evaluating the quality of text generation tasks using ROUGE score.

    Args:
        target_key (str): The key in the predicted batch that contains the target text.

    Methods:
        compute(predicted_batch): Computes the ROUGE score for the predicted batch.
        finalize(): Finalizes the ROUGE score computation and returns the average score.
    """

    def __init__(self, target_key: str = "references"):
        """
        Initialize the RougeScore metric with the target key and ROUGE evaluator.

        Args:
            target_key (str): Key for reference texts in the batch.
        """
        self.rouge = evaluate.load("rouge")
        self.target_key = target_key
        self.scores = {}

    def compute(self, predicted_batch: "PredictedBatch") -> None:
        """
        Compute the ROUGE score for the predicted batch.

        Args:
            predicted_batch (PredictedBatch): Batch containing predictions and reference data.
        """
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        scores = self.rouge.compute(predictions=generations, references=references)
        for key, value in scores.items():
            self.scores.setdefault(key, []).extend([float(value)] * len(generations))

    def finalize(self) -> dict:
        """
        Finalize the ROUGE score computation and return the average score.

        Returns:
            dict: Dictionary with the average ROUGE scores for each metric.
        """
        if not self.scores:
            return {key: 0.0 for key in self.rouge.metrics}
        out = {key: sum(values) / len(values) for key, values in self.scores.items()}
        self.score = {}
        return out


class MeteorScore(ValidationMetric):
    """
    MeteorScore is a validation metric for evaluating the quality of text generation tasks using METEOR score.

    Args:
        target_key (str): The key in the predicted batch that contains the target text.

    Methods:
        compute(predicted_batch): Computes the METEOR score for the predicted batch.
        finalize(): Finalizes the METEOR score computation and returns the average score.
    """

    def __init__(self, target_key: str = "references"):
        """
        Initialize the MeteorScore metric with the target key and METEOR evaluator.

        Args:
            target_key (str): Key for reference texts in the batch.
        """
        self.meteor = evaluate.load("meteor")
        self.target_key = target_key
        self.scores = []

    def compute(self, predicted_batch: "PredictedBatch") -> None:
        """
        Compute the METEOR score for the predicted batch.

        Args:
            predicted_batch (PredictedBatch): Batch containing predictions and reference data.
        """
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        scores = self.meteor.compute(predictions=generations, references=references)
        self.scores.extend([float(scores["meteor"])] * len(generations))

    def finalize(self) -> dict:
        """
        Finalize the METEOR score computation and return the average score.

        Returns:
            dict: Dictionary with the average METEOR score.
        """
        if not self.scores:
            return {"meteor": 0.0}
        out = {"meteor": sum(self.scores) / len(self.scores)}
        self.scores = []
        return out
