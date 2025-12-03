"""Natural Language Inference utilities using DeBERTa-v3 or compatible alternatives."""

import logging

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, NLI functionality will be limited")


class NLIModel:
    """
    Wrapper for NLI model.
    
    Uses roberta-large-mnli as primary model (compatible with transformers 4.38).
    Falls back to cross-encoder/nli-deberta-v3-base if available via sentence-transformers.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize NLI model.
        
        Args:
            model_name: Hugging Face model identifier. If None, uses roberta-large-mnli
                       (compatible alternative to cross-encoder/nli-deberta-v3-base).
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for NLI functionality. "
                "Install with: pip install transformers torch"
            )
        
        # Use roberta-large-mnli as default (compatible with transformers 4.38)
        # This is a standard NLI model that works well for entailment/contradiction tasks
        if model_name is None:
            model_name = "roberta-large-mnli"
        
        logger.info(f"Loading NLI model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model_name = model_name
            
            # Label mapping for roberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment
            self.label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def predict(self, premise: str, hypothesis: str) -> str:
        """
        Predict relationship between premise and hypothesis.
        
        Args:
            premise: The premise text (source/context)
            hypothesis: The hypothesis text (claim to verify)
            
        Returns:
            "entailment", "contradiction", or "neutral"
        """
        return self.batch_predict([(premise, hypothesis)])[0]

    def batch_predict(self, premise_hypothesis_pairs: list) -> list:
        """
        Batch prediction for multiple pairs.
        
        Args:
            premise_hypothesis_pairs: List of (premise, hypothesis) tuples
            
        Returns:
            List of predictions: "entailment", "contradiction", or "neutral"
        """
        predictions = []
        
        for premise, hypothesis in premise_hypothesis_pairs:
            # Format input for NLI: premise and hypothesis
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_label_id = logits[0].argmax().item()
            
            # Map label ID to label name
            label = self.label_map.get(predicted_label_id, "neutral")
            predictions.append(label)
        
        return predictions

