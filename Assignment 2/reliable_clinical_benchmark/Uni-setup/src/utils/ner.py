"""Named Entity Recognition using scispaCy."""

from typing import List, Set
import logging

logger = logging.getLogger(__name__)

try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore


class MedicalNER:
    """Medical NER using scispaCy."""

    def __init__(self, model_name: str = "en_core_sci_sm"):
        if spacy is None:
            logger.error(
                "spacy / scispaCy is not installed. "
                "Install it to enable MedicalNER utilities."
            )
            raise ImportError("spacy is required for MedicalNER but is not installed")

        logger.info(f"Loading scispaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(
                f"scispaCy model '{model_name}' not found. "
                "Install it with: python -m spacy download en_core_sci_sm"
            )
            raise

    def extract_entities(
        self,
        text: str,
        entity_types: List[str] = None,
    ) -> Set[str]:
        """Extract medical entities from text."""
        doc = self.nlp(text)

        entities = set()
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                entities.add(ent.text.lower())

        return entities

    def extract_clinical_entities(self, text: str) -> Set[str]:
        """Extract clinically relevant entities."""
        doc = self.nlp(text)
        entities = {ent.text.lower() for ent in doc.ents}
        return entities


