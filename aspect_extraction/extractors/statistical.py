"""Statistical aspect extractor using frequency and collocation analysis."""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from aspect_extraction.core.aspect import Aspect, Sentiment
from aspect_extraction.core.extractor import AspectExtractor


class StatisticalExtractor(AspectExtractor):
    """Statistical aspect extractor using frequency and collocation analysis.

    This extractor identifies aspects based on:
    - Term frequency analysis (frequent noun phrases)
    - Collocation detection (words that appear together)
    - PMI (Pointwise Mutual Information) scores
    - Context-based sentiment detection

    This approach works well for domain-specific text where aspects
    are mentioned frequently and consistently.

    Args:
        min_frequency: Minimum frequency for aspect candidates (default: 2)
        min_pmi: Minimum PMI score for collocations (default: 3.0)
        window_size: Context window size for collocations (default: 5)

    Example:
        >>> extractor = StatisticalExtractor(min_frequency=3)
        >>> texts = [
        ...     "The battery life is great",
        ...     "Battery life could be better",
        ...     "Love the battery life"
        ... ]
        >>> aspects = extractor.extract(texts[0])
    """

    def __init__(
        self,
        min_frequency: int = 2,
        min_pmi: float = 3.0,
        window_size: int = 5,
    ) -> None:
        """Initialize the statistical extractor.

        Args:
            min_frequency: Minimum occurrence count for aspects
            min_pmi: Minimum PMI score for significant collocations
            window_size: Window size for context analysis
        """
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        if min_pmi < 0:
            raise ValueError("min_pmi must be non-negative")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        self.min_frequency = min_frequency
        self.min_pmi = min_pmi
        self.window_size = window_size

        # Statistics gathered from training data
        self._noun_phrase_counts: Counter = Counter()
        self._collocation_counts: Counter = Counter()
        self._total_phrases = 0
        self._is_trained = False

        # Sentiment indicators
        self._sentiment_indicators = self._get_sentiment_indicators()

    def _get_sentiment_indicators(self) -> Dict[str, Sentiment]:
        """Get sentiment indicator words.

        Returns:
            Dictionary mapping words to sentiments
        """
        return {
            # Positive
            "excellent": Sentiment.POSITIVE,
            "great": Sentiment.POSITIVE,
            "good": Sentiment.POSITIVE,
            "amazing": Sentiment.POSITIVE,
            "love": Sentiment.POSITIVE,
            "perfect": Sentiment.POSITIVE,
            "best": Sentiment.POSITIVE,
            "fantastic": Sentiment.POSITIVE,
            "wonderful": Sentiment.POSITIVE,
            # Negative
            "poor": Sentiment.NEGATIVE,
            "bad": Sentiment.NEGATIVE,
            "terrible": Sentiment.NEGATIVE,
            "awful": Sentiment.NEGATIVE,
            "worst": Sentiment.NEGATIVE,
            "hate": Sentiment.NEGATIVE,
            "disappointing": Sentiment.NEGATIVE,
            "horrible": Sentiment.NEGATIVE,
        }

    def train(self, texts: List[str]) -> None:
        """Train the extractor on a corpus of texts.

        This builds frequency statistics used for aspect extraction.

        Args:
            texts: List of training texts

        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Training texts cannot be empty")

        self._noun_phrase_counts.clear()
        self._collocation_counts.clear()
        self._total_phrases = 0

        for text in texts:
            phrases = self._extract_noun_phrases(text)
            self._noun_phrase_counts.update(phrases)
            self._total_phrases += len(phrases)

            # Track collocations
            words = text.lower().split()
            for i in range(len(words) - 1):
                bigram = (words[i], words[i + 1])
                self._collocation_counts[bigram] += 1

        self._is_trained = True

    def extract(self, text: str) -> List[Aspect]:
        """Extract aspects from text using statistical analysis.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted aspects

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        aspects: List[Aspect] = []
        seen_texts: Set[str] = set()

        # Extract noun phrases as aspect candidates
        phrases = self._extract_noun_phrases(text)

        for phrase in phrases:
            # Skip if already processed
            if phrase.lower() in seen_texts:
                continue

            # Calculate confidence based on frequency (if trained)
            confidence = self._calculate_confidence(phrase)

            # Skip low-confidence aspects
            if confidence < 0.3:
                continue

            # Detect sentiment from context
            sentiment = self._detect_sentiment_in_context(text, phrase)

            # Find position in text
            start_pos = text.lower().find(phrase.lower())
            end_pos = start_pos + len(phrase) if start_pos >= 0 else None

            aspects.append(
                Aspect(
                    text=phrase,
                    sentiment=sentiment,
                    confidence=confidence,
                    start_pos=start_pos if start_pos >= 0 else None,
                    end_pos=end_pos,
                )
            )

            seen_texts.add(phrase.lower())

        return aspects

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases from text using simple heuristics.

        Args:
            text: Input text

        Returns:
            List of noun phrases
        """
        # Simple pattern-based extraction
        # In a real implementation, you might use spaCy or NLTK
        words = text.split()
        phrases = []

        # Common aspect patterns
        aspect_keywords = {
            "quality",
            "life",
            "performance",
            "battery",
            "camera",
            "screen",
            "display",
            "design",
            "price",
            "sound",
            "speed",
            "size",
            "build",
            "feature",
            "service",
            "support",
            "value",
        }

        for i, word in enumerate(words):
            word_clean = word.lower().strip(".,!?;:")

            # Single keyword
            if word_clean in aspect_keywords:
                phrases.append(word_clean)

            # Bigrams (adjective + keyword)
            if i > 0:
                prev_word = words[i - 1].lower().strip(".,!?;:")
                bigram = f"{prev_word} {word_clean}"

                # Check if second word is a keyword
                if word_clean in aspect_keywords:
                    phrases.append(bigram)

        return phrases

    def _calculate_confidence(self, phrase: str) -> float:
        """Calculate confidence score for an aspect candidate.

        Args:
            phrase: Aspect phrase

        Returns:
            Confidence score between 0 and 1
        """
        if not self._is_trained or self._total_phrases == 0:
            # Without training data, use base confidence
            return 0.5

        # Frequency-based confidence
        count = self._noun_phrase_counts.get(phrase, 0)

        if count < self.min_frequency:
            return 0.3

        # Normalize by total phrases
        frequency = count / self._total_phrases

        # Convert to confidence score (scaled)
        confidence = min(0.5 + (frequency * 100), 1.0)

        return confidence

    def _detect_sentiment_in_context(self, text: str, phrase: str) -> Optional[Sentiment]:
        """Detect sentiment of aspect from surrounding context.

        Args:
            text: Full text
            phrase: Aspect phrase

        Returns:
            Detected sentiment or None
        """
        # Find phrase position
        phrase_pos = text.lower().find(phrase.lower())
        if phrase_pos < 0:
            return None

        # Extract context window
        context_start = max(0, phrase_pos - 50)
        context_end = min(len(text), phrase_pos + len(phrase) + 50)
        context = text[context_start:context_end].lower()

        # Count sentiment words in context
        positive_count = 0
        negative_count = 0

        for word, sentiment in self._sentiment_indicators.items():
            if word in context:
                if sentiment == Sentiment.POSITIVE:
                    positive_count += 1
                elif sentiment == Sentiment.NEGATIVE:
                    negative_count += 1

        # Determine sentiment
        if positive_count > 0 and negative_count > 0:
            return Sentiment.MIXED
        elif positive_count > 0:
            return Sentiment.POSITIVE
        elif negative_count > 0:
            return Sentiment.NEGATIVE
        else:
            return None

    def calculate_pmi(self, word1: str, word2: str) -> float:
        """Calculate Pointwise Mutual Information for word pair.

        PMI measures how much more likely two words appear together
        compared to appearing independently.

        Args:
            word1: First word
            word2: Second word

        Returns:
            PMI score

        Raises:
            ValueError: If extractor is not trained
        """
        if not self._is_trained:
            raise ValueError("Extractor must be trained before calculating PMI")

        # Get counts
        bigram = (word1.lower(), word2.lower())
        joint_count = self._collocation_counts.get(bigram, 0)

        if joint_count == 0:
            return 0.0

        # Count individual words
        word1_count = sum(count for (w1, _), count in self._collocation_counts.items() if w1 == word1.lower())
        word2_count = sum(count for (_, w2), count in self._collocation_counts.items() if w2 == word2.lower())

        total_bigrams = sum(self._collocation_counts.values())

        if word1_count == 0 or word2_count == 0 or total_bigrams == 0:
            return 0.0

        # Calculate PMI
        p_xy = joint_count / total_bigrams
        p_x = word1_count / total_bigrams
        p_y = word2_count / total_bigrams

        pmi = np.log2(p_xy / (p_x * p_y))

        return float(pmi)

    def get_top_aspects(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the most frequent aspects from training data.

        Args:
            n: Number of top aspects to return

        Returns:
            List of (aspect, count) tuples

        Raises:
            ValueError: If extractor is not trained
        """
        if not self._is_trained:
            raise ValueError("Extractor must be trained first")

        return self._noun_phrase_counts.most_common(n)
