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
        return {"bleu": sum(self.scores) / len(self.scores)}


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
        scores = self.nist.compute(predictions=generations, references=references)
        self.scores.extend([scores["nist_mt"]] * len(generations))

    def finalize(self) -> dict:
        """
        Finalize the NIST score computation and return the average score.

        Returns:
            dict: Dictionary with the average NIST score.
        """
        if not self.scores:
            return {"nist": 0.0}
        return {"nist": sum(self.scores) / len(self.scores)}


class RougeScore(ValidationMetric):
    """
    RougeScore is a validation metric for evaluating the quality of text generation tasks using ROUGE score.

    Args:
        target_key (str): The key in the predicted batch that contains the target text.

    Methods:
        compute(predicted_batch): Computes the ROUGE score for the predicted batch.
        finalize(): Finalizes the ROUGE score computation and returns the average score.
    """

    def __init__(self, target_key: str = "references"):str = "references"):
        """ """
        Initialize the RougeScore metric with the target key and ROUGE evaluator.

        Args:rgs:
            target_key (str): Key for reference texts in the batch.
        """        """
        self.rouge = evaluate.load("rouge").rouge = evaluate.load("rouge")
        self.target_key = target_key
        self.scores = {}

    def compute(self, predicted_batch) -> None:    def compute(self, predicted_batch: "PredictedBatch") -> None:
        """atch
        Compute the ROUGE score for the predicted batch.

        Args:        references = predicted_batch.reference_data[self.target_key]
            predicted_batch (PredictedBatch): Batch containing predictions and reference data.s = self.rouge.compute(predictions=generations, references=references)
        """
        generations = predicted_batch.prediction self.scores.setdefault(key, []).extend([float(value)] * len(generations))
        references = predicted_batch.reference_data[self.target_key]
        scores = self.rouge.compute(predictions=generations, references=references)
        for key, value in scores.items():
            self.scores.setdefault(key, []).extend([float(value)] * len(generations))        Finalize the ROUGE score computation and return the average score.

    def finalize(self) -> dict:urns:
        """es for each metric.
        Finalize the ROUGE score computation and return the average score.        """
t self.scores:
        Returns:
            dict: Dictionary with the average ROUGE scores for each metric.urn {key: sum(values) / len(values) for key, values in self.scores.items()}
        """
        if not self.scores:
            return {key: 0.0 for key in self.rouge.metrics}
        return {key: sum(values) / len(values) for key, values in self.scores.items()}
n tasks using METEOR score.

class MeteorScore(ValidationMetric):
    """        target_key (str): The key in the predicted batch that contains the target text.
    MeteorScore is a validation metric for evaluating the quality of text generation tasks using METEOR score.
:
    Args:ted batch.
        target_key (str): The key in the predicted batch that contains the target text.        finalize(): Finalizes the METEOR score computation and returns the average score.

    Methods:
        compute(predicted_batch): Computes the METEOR score for the predicted batch.nit__(self, target_key: str = "references"):
        finalize(): Finalizes the METEOR score computation and returns the average score.
    """ey and METEOR evaluator.

    def __init__(self, target_key: str = "references"):        Args:
        """            target_key (str): Key for reference texts in the batch.
        Initialize the MeteorScore metric with the target key and METEOR evaluator.
 self.meteor = evaluate.load("meteor")
        Args:
            target_key (str): Key for reference texts in the batch.        self.scores = []
        """
        self.meteor = evaluate.load("meteor")
        self.target_key = target_key        from nnt.validators.validator import PredictedBatch
        self.scores = []

    def compute(self, predicted_batch) -> None:
        """ scores = self.meteor.compute(predictions=generations, references=references)
        Compute the METEOR score for the predicted batch.        self.scores.extend([float(scores["meteor"])] * len(generations))

        Args:alize(self) -> dict:
            predicted_batch (PredictedBatch): Batch containing predictions and reference data.
        """        Finalize the METEOR score computation and return the average score.
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        scores = self.meteor.compute(predictions=generations, references=references) dict: Dictionary with the average METEOR score.
        self.scores.extend([float(scores["meteor"])] * len(generations))

    def finalize(self) -> dict:eor": 0.0}
        """        return {"meteor": sum(self.scores) / len(self.scores)}
        Finalize the METEOR score computation and return the average score.        Returns:            dict: Dictionary with the average METEOR score.        """        if not self.scores:            return {"meteor": 0.0}        return {"meteor": sum(self.scores) / len(self.scores)}