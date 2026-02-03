"""Natural Language Inference utilities using DeBERTa-v3."""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAX_LENGTH = 512


class NLIModel:
    """Wrapper for NLI model (DeBERTa-v3)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        *,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ):
        logger.info(f"Loading NLI model: {model_name}")
        self.max_length = int(max_length)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, premise: str, hypothesis: str) -> str:
        """Predict relationship between premise and hypothesis."""
        import torch

        inputs = self.tokenizer(
            str(premise or ""),
            str(hypothesis or ""),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        pred_id = int(torch.argmax(logits, dim=-1).item())
        label = str(self.model.config.id2label.get(pred_id, "")).lower()
        if "entail" in label:
            return "entailment"
        elif "contradict" in label:
            return "contradiction"
        else:
            return "neutral"

    def batch_predict(self, premise_hypothesis_pairs: list) -> list:
        """Batch prediction for multiple pairs."""
        import torch

        pairs: List[Tuple[str, str]] = [
            (str(p or ""), str(h or "")) for p, h in premise_hypothesis_pairs
        ]
        if not pairs:
            return []

        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]

        inputs = self.tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1).tolist()
        out: List[str] = []
        for pred_id in pred_ids:
            label = str(self.model.config.id2label.get(int(pred_id), "")).lower()
            if "entail" in label:
                out.append("entailment")
            elif "contradict" in label:
                out.append("contradiction")
            else:
                out.append("neutral")

        return out

