"""Natural Language Inference utilities using DeBERTa-v3."""

from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class NLIModel:
    """Wrapper for NLI model (DeBERTa-v3)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        logger.info(f"Loading NLI model: {model_name}")
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=-1,
        )

    def predict(self, premise: str, hypothesis: str) -> str:
        """Predict relationship between premise and hypothesis."""
        input_text = f"{premise} [SEP] {hypothesis}"
        result = self.classifier(input_text)[0]

        label = result["label"].lower()
        if "entail" in label:
            return "entailment"
        elif "contradict" in label:
            return "contradiction"
        else:
            return "neutral"

    def batch_predict(self, premise_hypothesis_pairs: list) -> list:
        """Batch prediction for multiple pairs."""
        inputs = [f"{p} [SEP] {h}" for p, h in premise_hypothesis_pairs]
        results = self.classifier(inputs)

        predictions = []
        for result in results:
            label = result["label"].lower()
            if "entail" in label:
                predictions.append("entailment")
            elif "contradict" in label:
                predictions.append("contradiction")
            else:
                predictions.append("neutral")

        return predictions


