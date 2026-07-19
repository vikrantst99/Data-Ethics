"""
RAG retrieval over the RAI policy catalogue (the "Semantik" step).

Given a query (built from the proposal's trigger words), find the most relevant
RAI policy text. We try a real semantic vector store first:

    HuggingFace embeddings + Chroma  ->  semantic similarity

If those packages (or the embedding model download) are not available, we fall
back to a small TF-IDF cosine search over the same nine policy texts. Either
way search(query, k) returns the top-k policies, so the rest of the agent never
breaks.
"""

import uuid

from .policies import policy_docs


class PolicyRetriever:
    def __init__(self, prefer_semantic=True):
        self.docs = policy_docs()                 # [(policy_id, text), ...]
        self.ids = [d[0] for d in self.docs]
        self.texts = [d[1] for d in self.docs]
        self.mode = "keyword"
        self._store = None
        self._tfidf = None
        self._matrix = None
        if prefer_semantic:
            self._try_semantic()
        if self.mode != "semantic":
            self._build_keyword()

    def _try_semantic(self):
        # Build a Chroma vector store if the optional packages are installed
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
        except Exception:
            return
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
            documents = []
            for pid, text in self.docs:
                documents.append(Document(page_content=text,
                                          metadata={"policy_id": pid}))
            # A unique collection name per instance keeps repeated rebuilds from appending the same docs to one shared collection
            self._store = Chroma.from_documents(
                documents, embeddings,
                collection_name="rai_policies_" + uuid.uuid4().hex)
            self.mode = "semantic"
        except Exception:
            # Any failure (e.g. no network to download the model) -> keyword.
            self._store = None

    def _build_keyword(self):
        # TF-IDF cosine fallback over the nine policy texts.
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(stop_words="english")
        self._matrix = self._tfidf.fit_transform(self.texts)

    def search(self, query, k=3):
        # Return the best k policies as dicts, keeping only the first hit per policy so none appears twice
        if self.mode == "semantic":
            raw = self._search_semantic(query, k + len(self.ids))
        else:
            raw = self._search_keyword(query, k)
        seen = set()
        unique = []
        for hit in raw:
            if hit["policy_id"] in seen:
                continue
            seen.add(hit["policy_id"])
            unique.append(hit)
        return unique[:k]

    def _search_semantic(self, query, k):
        hits = self._store.similarity_search_with_score(query, k=k)
        results = []
        for doc, distance in hits:
            results.append({
                "policy_id": doc.metadata.get("policy_id", "?"),
                "text": doc.page_content,
                # Chroma returns a distance; smaller is closer. Flip to a score
                "score": round(1.0 / (1.0 + float(distance)), 4),
            })
        return results

    def _search_keyword(self, query, k):
        from sklearn.metrics.pairwise import cosine_similarity
        q = self._tfidf.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = sims.argsort()[::-1][:k]
        results = []
        for i in order:
            results.append({
                "policy_id": self.ids[i],
                "text": self.texts[i],
                "score": round(float(sims[i]), 4),
            })
        return results
