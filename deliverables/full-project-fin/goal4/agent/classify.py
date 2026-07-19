"""
The Goal-2 model as the agent's gatekeeper, plus XAI trigger words.

The Classifier wraps the loaded Goal-2 pipeline. It gives three things:
- proba(text)         : the Red-Flag probability
- predict(text)       : a hard 0/1 decision at a fixed threshold (default 0.40)
- trigger_words(text) : the words that pushed the decision (local XAI grounding)

The trigger-word logic is the same tfidf * coefficient view used in Goal 3. It
replaces the old, missing 'xai2.explain_prediction' from the Version-1 agent.
"""

# Version-2 model is a plain pipeline, so we apply the 0.40 threshold ourselves
DEFAULT_THRESHOLD = 0.40


class Classifier:
    def __init__(self, pipeline, threshold=DEFAULT_THRESHOLD):
        self.pipeline = pipeline
        self.threshold = threshold
        self.vectorizer = pipeline.named_steps["tfidf"]
        self.clf = pipeline.named_steps["clf"]

    def proba(self, text):
        # Red-Flag probability for a single proposal text
        return float(self.pipeline.predict_proba([text])[0, 1])

    def predict(self, text, threshold=None):
        # Hard decision: 1 = Red Flag, 0 = Compliant, using the threshold
        if threshold is None:
            threshold = self.threshold
        return int(self.proba(text) >= threshold)

    def trigger_words(self, text, n=10):
        # Local explanation: contribution = tfidf value * coefficient per word
        if not hasattr(self.clf, "coef_"):
            raise AttributeError("Trigger words need a linear model (coef_).")
        values = self.vectorizer.transform([text]).toarray()[0]
        coefficients = self.clf.coef_[0]
        names = self.vectorizer.get_feature_names_out()

        pairs = []
        for i in range(len(values)):
            if values[i] != 0:
                contribution = float(values[i] * coefficients[i])
                pairs.append((names[i], contribution))
        pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return pairs[:n]
