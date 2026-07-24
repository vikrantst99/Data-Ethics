import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import spacy
import joblib

nlp = spacy.load("en_core_web_sm")
remove_pos = {"DET", "CCONJ", "PUNCT", "SPACE"}


def load_data(x):
    df = pd.read_csv(x)
    df["y"] = (df["label"] == "Red Flag").astype(int)
    return df

def train_test_splitter(df):
    idx_train, idx_test = train_test_split(df.index, stratify=df["y"], random_state=42, test_size=0.2)
    X_train = df.loc[idx_train, "description"].fillna("")
    X_test = df.loc[idx_test, "description"].fillna("")
    y_train = df.loc[idx_train, "y"]
    y_test = df.loc[idx_test, "y"]
    return X_train, X_test, y_train, y_test


class ThresholdAdjustor(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, threshold=0.4):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        proba = self.estimator.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    @property
    def coef_(self):
        return self.estimator.coef_

    @property
    def intercept_(self):
        return self.estimator.intercept_


def token_pos(text):
    doc = nlp(text.lower())
    tokens = []
    for i in doc:
        if i.pos_ not in remove_pos and not i.is_space and len(i.text) > 1:
            tokens.append(i.text)
    return tokens


def pipeLine(ngram_range=(1, 2), min_df=2, max_df=0.9, threshold=0.4):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            tokenizer=token_pos,
            token_pattern=None,
            sublinear_tf=True,
        )),
        ("clf", ThresholdAdjustor(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            threshold=threshold,
        )),
    ])
    return pipeline

def explain_prediction(pipeline, text, n=10):
    vec=pipeline.named_steps["tfidf"]
    clf=pipeline.named_steps["clf"]

    tfidf_scores=vec.transform([text]).toarray()[0]
    weights=clf.coef_[0]
    words=vec.get_feature_names_out()

    results=[]

    for i in range(len(words)):
        if tfidf_scores[i] != 0:
            word=words[i]
            score=tfidf_scores[i]*weights[i]
            results.append((word,score))
    results.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return results[:n]

def audit_trail(pipeline, text, y_true, y_pred, proba, n_terms=10):
    
    # Get the top trigger words for this specific proposal
    # explain_prediction returns a list of (word, score) tuples
    top_words=explain_prediction(pipeline, text, n_terms)

    # Separate Red Flag drivers from Compliant drivers
    # A positive score means the word pushed toward Red Flag
    # A negative score means the word pushed toward Compliant
    trigger_words=[word for word, score in top_words if score>0]
    compliant_words=[word for word, score in top_words if score<0]

    # Build the audit log as a dictionary
    # Every key is one piece of information an auditor needs to reconstruct this decision
    log={
        "actual":int(y_true),                          # what the true label was
        "predicted":int(y_pred),                       # what the model predicted
        "probability":round(float(proba),4),           # how confident the model was
        "correct":bool(y_true==y_pred),                # whether the prediction was right
        "trigger_words":trigger_words,                 # words that pushed toward Red Flag
        "explanation_method":"LogReg coef_ x tfidf",  # always deterministic, never approximate
    }
    return log


if __name__ == "__main__":
    df = load_data(r"C:\MyGithubSpace\Data-Ethics\data\proposals_1000_EN 1.csv")
    X_train, X_test, y_train, y_test = train_test_splitter(df)

    pipeline = pipeLine(ngram_range=(1, 2), min_df=2, threshold=0.4)
    pipeline.fit(X_train, y_train)
    print("Pipeline trained")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    joblib.dump(pipeline, r"C:\MyGithubSpace\Data-Ethics\Works\Week 4\logRegpipelineV2.pkl")
    print("Saved logRegpipelineV2.pkl")