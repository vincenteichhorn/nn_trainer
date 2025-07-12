from nnt.validation_metrics.validation_metric import ValidationMetric
import evaluate


class BleuScore(ValidationMetric):
    """
    BleuScore is a validation metric for evaluating the quality of text generation tasks using BLEU score.

    Args:
        n_gram (int): The maximum n-gram size to consider for BLEU score calculation.
        smooth (bool): Whether to apply smoothing to the BLEU score calculation.

    Methods:
        compute(predicted_batch, target_batch): Computes the BLEU score for the predicted batch against the target batch.
    """

    def __init__(self, target_key="references"):
        self.bleu = evaluate.load("bleu")
        self.target_key = target_key
        self.scores = []

    def compute(self, predicted_batch):
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        scores = self.bleu.compute(predictions=generations, references=references)
        self.scores.extend([scores["bleu"]] * len(generations))

    def finalize(self):
        """
        Finalize the BLEU score computation and return the average score.
        """
        if not self.scores:
            return {"bleu": 0.0}
        return {"bleu": sum(self.scores) / len(self.scores)}


class NistScore(ValidationMetric):
    """
    NistScore is a validation metric for evaluating the quality of text generation tasks using NIST score.

    Args:
        n_gram (int): The maximum n-gram size to consider for NIST score calculation.
        smooth (bool): Whether to apply smoothing to the NIST score calculation.

    Methods:
        compute(predicted_batch, target_batch): Computes the NIST score for the predicted batch against the target batch.
    """

    def __init__(self, target_key="references"):
        self.nist = evaluate.load("nist_mt")
        self.target_key = target_key
        self.scores = []

    def compute(self, predicted_batch):
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        scores = self.nist.compute(predictions=generations, references=references)
        self.scores.extend([scores["nist_mt"]] * len(generations))

    def finalize(self):
        """
        Finalize the NIST score computation and return the average score.
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

    def __init__(self, target_key="references"):
        self.rouge = evaluate.load("rouge")
        self.target_key = target_key
        self.scores = {}

    def compute(self, predicted_batch):
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        scores = self.rouge.compute(predictions=generations, references=references)
        for key, value in scores.items():
            self.scores.setdefault(key, []).extend([float(value)] * len(generations))

    def finalize(self):
        """
        Finalize the ROUGE score computation and return the average score.
        """
        if not self.scores:
            return {key: 0.0 for key in self.rouge.metrics}
        return {key: sum(values) / len(values) for key, values in self.scores.items()}


class MeteorScore(ValidationMetric):
    """
    MeteorScore is a validation metric for evaluating the quality of text generation tasks using METEOR score.

    Args:
        target_key (str): The key in the predicted batch that contains the target text.

    Methods:
        compute(predicted_batch): Computes the METEOR score for the predicted batch.
        finalize(): Finalizes the METEOR score computation and returns the average score.
    """

    def __init__(self, target_key="references"):
        self.meteor = evaluate.load("meteor")
        self.target_key = target_key
        self.scores = []

    def compute(self, predicted_batch):
        generations = predicted_batch.prediction
        references = predicted_batch.reference_data[self.target_key]
        scores = self.meteor.compute(predictions=generations, references=references)
        self.scores.extend([float(scores["meteor"])] * len(generations))

    def finalize(self):
        """
        Finalize the METEOR score computation and return the average score.
        """
        if not self.scores:
            return {"meteor": 0.0}
        return {"meteor": sum(self.scores) / len(self.scores)}
