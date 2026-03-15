from __future__ import annotations

import re
import importlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


TEXT_COLUMNS = ["Synopsis", "Genre", "Tags"]


def load_dataset(data_path: str = "data/kdrama_list_cleaned.csv") -> pd.DataFrame:
    """Load dataset from data directory with a fallback to project root."""
    path = Path(data_path)
    if not path.exists():
        fallback = Path("kdrama_list_cleaned.csv")
        if not fallback.exists():
            raise FileNotFoundError("Could not find dataset in data/ or project root.")
        path = fallback

    df = pd.read_csv(path)
    return standardize_columns(df)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and fill required fields."""
    rename_map = {
        "img URL": "img_URL",
        "Img URL": "img_URL",
        "Image URL": "img_URL",
    }
    df = df.rename(columns=rename_map).copy()

    required = ["Title", "Year", "Score", "Synopsis", "Genre", "Tags", "img_URL"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    for col in ["Title", "Synopsis", "Genre", "Tags", "img_URL"]:
        df[col] = df[col].fillna("").astype(str)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    df = df.drop_duplicates(subset=["Title"]).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """Lowercase and keep alphanumeric tokens for stable vectorization."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_feature_column(df: pd.DataFrame, text_columns: Sequence[str] = TEXT_COLUMNS) -> pd.DataFrame:
    """Create a combined text feature from synopsis, genre, and tags."""
    temp = df.copy()
    for col in text_columns:
        if col not in temp.columns:
            temp[col] = ""
        temp[col] = temp[col].fillna("").astype(str).map(clean_text)

    temp["combined_text"] = temp[list(text_columns)].agg(" ".join, axis=1).str.strip()
    return temp


def build_tfidf_model(
    texts: Iterable[str],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 15000,
):
    """Fit a TF-IDF vectorizer and return vectorizer and sparse matrix."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def recommend_with_tfidf(
    title: str,
    df: pd.DataFrame,
    tfidf_matrix,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return top-K recommendations using cosine similarity over TF-IDF vectors."""
    idx = find_title_index(df, title)
    sims = linear_kernel(tfidf_matrix[idx : idx + 1], tfidf_matrix).flatten()
    return format_recommendations(df, sims, idx, top_k)


def build_sentence_embeddings(texts: Sequence[str], model_name: str = "all-MiniLM-L6-v2"):
    """Encode text with a SentenceTransformer model.

    Import is local so users can still run TF-IDF mode without installing embeddings dependencies.
    """
    try:
        sentence_transformers_module = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(sentence_transformers_module, "SentenceTransformer")
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Install it from requirements.txt."
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(texts), show_progress_bar=False)
    return model, np.asarray(embeddings)


def recommend_with_embeddings(
    title: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return top-K recommendations using cosine similarity over sentence embeddings."""
    idx = find_title_index(df, title)
    sims = cosine_similarity(embeddings[idx : idx + 1], embeddings).flatten()
    return format_recommendations(df, sims, idx, top_k)


def find_title_index(df: pd.DataFrame, title: str) -> int:
    mask = df["Title"].str.lower().str.strip() == title.lower().strip()
    matches = df.index[mask]
    if len(matches) == 0:
        raise ValueError(f"Title '{title}' not found in dataset.")
    return int(matches[0])


def format_recommendations(df: pd.DataFrame, similarities: np.ndarray, src_idx: int, top_k: int) -> pd.DataFrame:
    similarities = similarities.copy()
    similarities[src_idx] = -1
    top_indices = similarities.argsort()[-top_k:][::-1]

    cols = ["Title", "Year", "Score", "Genre", "Synopsis", "img_URL"]
    recs = df.iloc[top_indices][cols].copy()
    recs["similarity"] = similarities[top_indices]
    return recs.reset_index(drop=True)


def build_proxy_relevance(df: pd.DataFrame) -> Dict[int, set]:
    """Create weak ground-truth relevance using shared genre and tags.

    This allows offline ranking metrics when explicit user interaction labels are unavailable.
    """
    genre_sets = df["Genre"].map(_split_tokens)
    tag_sets = df["Tags"].map(_split_tokens)

    relevance: Dict[int, set] = {}
    for i in range(len(df)):
        target_genres = genre_sets.iloc[i]
        target_tags = tag_sets.iloc[i]
        rel = set()
        for j in range(len(df)):
            if i == j:
                continue
            genre_overlap = len(target_genres.intersection(genre_sets.iloc[j])) > 0
            tag_overlap = len(target_tags.intersection(tag_sets.iloc[j])) > 0
            if genre_overlap or tag_overlap:
                rel.add(j)
        relevance[i] = rel
    return relevance


def evaluate_ranking(
    df: pd.DataFrame,
    recommend_fn: Callable[[str, int], pd.DataFrame],
    k: int = 10,
    sample_size: int = 200,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate a recommender with Precision@K, Recall@K, and MRR.

    recommend_fn must accept (title, top_k) and return a DataFrame that includes Title.
    """
    rng = np.random.default_rng(random_state)
    relevance = build_proxy_relevance(df)

    all_indices = np.arange(len(df))
    if sample_size < len(df):
        eval_indices = rng.choice(all_indices, size=sample_size, replace=False)
    else:
        eval_indices = all_indices

    precision_scores: List[float] = []
    recall_scores: List[float] = []
    reciprocal_ranks: List[float] = []

    title_to_idx = {title: idx for idx, title in enumerate(df["Title"]) }

    for idx in eval_indices:
        title = df.iloc[idx]["Title"]
        relevant = relevance.get(int(idx), set())
        if not relevant:
            continue

        preds_df = recommend_fn(title, k)
        predicted_titles = preds_df["Title"].tolist()
        predicted_indices = [title_to_idx[t] for t in predicted_titles if t in title_to_idx]

        precision_scores.append(precision_at_k(predicted_indices, relevant, k))
        recall_scores.append(recall_at_k(predicted_indices, relevant, k))
        reciprocal_ranks.append(reciprocal_rank(predicted_indices, relevant, k))

    if not precision_scores:
        return {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0}

    return {
        "precision@k": float(np.mean(precision_scores)),
        "recall@k": float(np.mean(recall_scores)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def precision_at_k(predicted: Sequence[int], relevant: set, k: int) -> float:
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(predicted: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(predicted: Sequence[int], relevant: set, k: int) -> float:
    for rank, item in enumerate(predicted[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def _split_tokens(text: str) -> set:
    if not isinstance(text, str):
        return set()
    tokens = [t.strip().lower() for t in re.split(r"[,|/]", text) if t.strip()]
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return set(tokens)
