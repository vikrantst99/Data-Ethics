"""Simple baselines without machine learning, used only for comparison"""

import numpy as np

# Words and phrases that often appear in a rule violation.

RULE_TERMS = [
    "social scoring", "without consent", "without user consent", "emotion recognition",
    "biometric", "no human oversight", "fully autonomous", "no explanation",
    "black box", "manipulat", "vulnerable", "subliminal", "track users",
    "store all", "indefinitely", "share data with third", "without informing",
]


def rule_predict(texts, terms=RULE_TERMS):
    # Return 1 for every text that contains one of the terms, else 0
    predictions = []
    for text in texts:
        text_lower = str(text).lower()
        found = 0
        for term in terms:
            if term.lower() in text_lower:
                found = 1
                break
        predictions.append(found)
    return np.array(predictions, dtype=int)


def majority_predict(n, majority_class=0):
    # Always predict the same class (by default 0 = Compliant)
    return np.full(int(n), majority_class, dtype=int)
