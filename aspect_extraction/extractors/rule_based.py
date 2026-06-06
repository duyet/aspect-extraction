"""Rule-based aspect extractor using linguistic patterns."""

import re
from typing import Dict, List, Optional, Set

from aspect_extraction.core.aspect import Aspect, Sentiment
from aspect_extraction.core.extractor import AspectExtractor


class RuleBasedExtractor(AspectExtractor):
    """Rule-based aspect extractor using dependency parsing and POS patterns.

    This extractor uses linguistic rules to identify aspects based on:
    - Noun phrases (potential aspects)
    - Dependency relations (subject, object, modifier)
    - POS tag patterns (e.g., ADJ + NOUN)
    - Sentiment lexicons for polarity detection

    The advantage of this approach is speed and interpretability,
    though it may have lower recall than ML-based methods.

    Args:
        use_spacy: Whether to use spaCy for advanced parsing (default: True)
        sentiment_lexicon: Optional custom sentiment lexicon

    Example:
        >>> extractor = RuleBasedExtractor()
        >>> text = "The camera quality is excellent but battery life is poor"
        >>> aspects = extractor.extract(text)
        >>> for aspect in aspects:
        ...     print(f"{aspect.text}: {aspect.sentiment}")
        camera quality: positive
        battery life: negative
    """

    def __init__(
        self,
        use_spacy: bool = True,
        sentiment_lexicon: Optional[Dict[str, Sentiment]] = None,
    ) -> None:
        """Initialize the rule-based extractor.

        Args:
            use_spacy: Whether to use spaCy for NLP processing
            sentiment_lexicon: Custom sentiment words mapping
        """
        self.use_spacy = use_spacy
        self._nlp = None

        # Default sentiment lexicon
        self._sentiment_lexicon = sentiment_lexicon or self._get_default_sentiment_lexicon()

        if use_spacy:
            try:
                import spacy

                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not downloaded, fall back to non-spacy mode
                    self.use_spacy = False
            except ImportError:
                # spaCy not installed
                self.use_spacy = False

    def _get_default_sentiment_lexicon(self) -> Dict[str, Sentiment]:
        """Get default sentiment lexicon.

        Returns:
            Dictionary mapping words to sentiment polarities
        """
        return {
            # Positive words
            "excellent": Sentiment.POSITIVE,
            "great": Sentiment.POSITIVE,
            "good": Sentiment.POSITIVE,
            "amazing": Sentiment.POSITIVE,
            "wonderful": Sentiment.POSITIVE,
            "fantastic": Sentiment.POSITIVE,
            "perfect": Sentiment.POSITIVE,
            "love": Sentiment.POSITIVE,
            "best": Sentiment.POSITIVE,
            "outstanding": Sentiment.POSITIVE,
            "superb": Sentiment.POSITIVE,
            "awesome": Sentiment.POSITIVE,
            "brilliant": Sentiment.POSITIVE,
            # Negative words
            "poor": Sentiment.NEGATIVE,
            "bad": Sentiment.NEGATIVE,
            "terrible": Sentiment.NEGATIVE,
            "awful": Sentiment.NEGATIVE,
            "horrible": Sentiment.NEGATIVE,
            "worst": Sentiment.NEGATIVE,
            "disappointing": Sentiment.NEGATIVE,
            "disappoints": Sentiment.NEGATIVE,
            "hate": Sentiment.NEGATIVE,
            "useless": Sentiment.NEGATIVE,
            "mediocre": Sentiment.NEGATIVE,
            "subpar": Sentiment.NEGATIVE,
        }

    def extract(self, text: str) -> List[Aspect]:
        """Extract aspects from text using rule-based patterns.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted aspects with sentiment

        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if self.use_spacy and self._nlp is not None:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)

    def _extract_with_spacy(self, text: str) -> List[Aspect]:
        """Extract aspects using spaCy's dependency parsing.

        Args:
            text: Input text

        Returns:
            List of extracted aspects
        """
        if self._nlp is None:
            return []

        doc = self._nlp(text)
        aspects: List[Aspect] = []
        seen_texts: Set[str] = set()

        for token in doc:
            # Look for nouns that could be aspects
            if token.pos_ in ("NOUN", "PROPN"):
                # Get the noun phrase
                aspect_text = self._get_noun_phrase(token)

                # Skip if already seen
                if aspect_text.lower() in seen_texts:
                    continue

                # Detect sentiment from context
                sentiment = self._detect_sentiment_spacy(token, doc)

                # Calculate confidence based on POS patterns
                confidence = self._calculate_confidence_spacy(token)

                aspects.append(
                    Aspect(
                        text=aspect_text,
                        sentiment=sentiment,
                        confidence=confidence,
                        start_pos=token.idx,
                        end_pos=token.idx + len(aspect_text),
                    )
                )

                seen_texts.add(aspect_text.lower())

        return aspects

    def _get_noun_phrase(self, token) -> str:  # type: ignore
        """Extract noun phrase including modifiers.

        Args:
            token: spaCy token

        Returns:
            Complete noun phrase text
        """
        # Include adjective modifiers
        phrase_tokens = []

        # Add modifiers before the noun
        for child in token.lefts:
            if child.dep_ in ("amod", "compound", "nummod"):
                phrase_tokens.append(child.text)

        # Add the noun itself
        phrase_tokens.append(token.text)

        # Add modifiers after the noun
        for child in token.rights:
            if child.dep_ in ("compound", "nn"):
                phrase_tokens.append(child.text)

        return " ".join(phrase_tokens)

    def _detect_sentiment_spacy(self, token, doc) -> Optional[Sentiment]:  # type: ignore
        """Detect sentiment using dependency relations and lexicon.

        Args:
            token: Target token (aspect)
            doc: spaCy document

        Returns:
            Detected sentiment or None
        """
        # Check for sentiment words in the vicinity
        sentiment_words = []

        # Look at the token's head and children
        for related_token in list(token.children) + [token.head]:
            if related_token.text.lower() in self._sentiment_lexicon:
                sentiment_words.append(self._sentiment_lexicon[related_token.text.lower()])

        # Look for sentiment words in nearby adjectives
        for other_token in doc:
            if (
                other_token.pos_ == "ADJ"
                and abs(other_token.i - token.i) <= 3
                and other_token.text.lower() in self._sentiment_lexicon
            ):
                sentiment_words.append(self._sentiment_lexicon[other_token.text.lower()])

        # Determine overall sentiment
        if not sentiment_words:
            return None

        positive_count = sum(1 for s in sentiment_words if s == Sentiment.POSITIVE)
        negative_count = sum(1 for s in sentiment_words if s == Sentiment.NEGATIVE)

        if positive_count > 0 and negative_count > 0:
            return Sentiment.MIXED
        elif positive_count > 0:
            return Sentiment.POSITIVE
        elif negative_count > 0:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.NEUTRAL

    def _calculate_confidence_spacy(self, token) -> float:  # type: ignore
        """Calculate confidence score based on linguistic features.

        Args:
            token: spaCy token

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.7  # Base confidence

        # Higher confidence for nouns with modifiers
        if any(child.dep_ in ("amod", "compound") for child in token.children):
            confidence += 0.15

        # Higher confidence for subjects and objects
        if token.dep_ in ("nsubj", "dobj", "pobj"):
            confidence += 0.15

        return min(confidence, 1.0)

    def _extract_with_patterns(self, text: str) -> List[Aspect]:
        """Extract aspects using simple regex patterns (fallback).

        Args:
            text: Input text

        Returns:
            List of extracted aspects
        """
        aspects: List[Aspect] = []

        # Simple pattern: adjective + noun (e.g., "battery life", "camera quality")
        pattern = r"\b([a-zA-Z]+\s+(?:quality|life|performance|battery|camera|screen|design|price|sound|speed|size))\b"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            aspect_text = match.group(1)

            # Detect sentiment from surrounding words
            context_start = max(0, match.start() - 30)
            context_end = min(len(text), match.end() + 30)
            context = text[context_start:context_end]

            sentiment = self._detect_sentiment_pattern(context)

            aspects.append(
                Aspect(
                    text=aspect_text,
                    sentiment=sentiment,
                    confidence=0.6,  # Lower confidence for pattern-based
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=context,
                )
            )

        return aspects

    def _detect_sentiment_pattern(self, context: str) -> Optional[Sentiment]:
        """Detect sentiment from context using lexicon.

        Args:
            context: Context text around aspect

        Returns:
            Detected sentiment or None
        """
        context_lower = context.lower()

        positive_count = sum(
            1
            for word in self._sentiment_lexicon
            if self._sentiment_lexicon[word] == Sentiment.POSITIVE and word in context_lower
        )

        negative_count = sum(
            1
            for word in self._sentiment_lexicon
            if self._sentiment_lexicon[word] == Sentiment.NEGATIVE and word in context_lower
        )

        if positive_count > 0 and negative_count > 0:
            return Sentiment.MIXED
        elif positive_count > 0:
            return Sentiment.POSITIVE
        elif negative_count > 0:
            return Sentiment.NEGATIVE
        else:
            return None
