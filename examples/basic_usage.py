"""Basic usage examples for aspect extraction."""

from aspect_extraction import extract_aspects
from aspect_extraction.core.factory import create_extractor

# Example 1: Simple extraction using the convenience function
print("=" * 60)
print("Example 1: Simple Extraction")
print("=" * 60)

text = "The camera quality is excellent but battery life is poor"
aspects = extract_aspects(text, method="rule")

print(f"\nText: {text}\n")
print("Extracted aspects:")
for aspect in aspects:
    print(f"  - {aspect.text}: {aspect.sentiment}")

# Example 2: Using an extractor instance for multiple texts
print("\n" + "=" * 60)
print("Example 2: Batch Processing")
print("=" * 60)

extractor = create_extractor(method="rule")

texts = [
    "The food quality is great and service is outstanding",
    "Terrible camera quality and slow performance",
    "Amazing screen display but poor battery life",
]

for i, text in enumerate(texts, 1):
    aspects = extractor.extract(text)
    print(f"\nText {i}: {text}")
    print("Aspects:")
    for aspect in aspects:
        sentiment = aspect.sentiment.value if aspect.sentiment else "neutral"
        print(f"  - {aspect.text} [{sentiment}, confidence: {aspect.confidence:.2f}]")

# Example 3: Filtering by confidence
print("\n" + "=" * 60)
print("Example 3: Confidence Filtering")
print("=" * 60)

text = "The screen quality is good, camera is amazing, and price is fair"
all_aspects = extract_aspects(text, method="rule")
high_confidence = [a for a in all_aspects if a.confidence >= 0.8]

print(f"\nText: {text}\n")
print(f"All aspects ({len(all_aspects)}):")
for aspect in all_aspects:
    print(f"  - {aspect.text} (confidence: {aspect.confidence:.2f})")

print(f"\nHigh confidence aspects ({len(high_confidence)}):")
for aspect in high_confidence:
    print(f"  - {aspect.text} (confidence: {aspect.confidence:.2f})")

# Example 4: Different extraction methods
print("\n" + "=" * 60)
print("Example 4: Comparing Methods")
print("=" * 60)

text = "Great battery life and excellent camera quality"

for method in ["rule", "statistical"]:
    try:
        aspects = extract_aspects(text, method=method)
        print(f"\n{method.upper()} method:")
        for aspect in aspects:
            print(f"  - {aspect.text}")
    except Exception as e:
        print(f"\n{method.upper()} method: {e}")

print("\n" + "=" * 60)
