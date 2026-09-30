"""Content-based movie recommender (TF-IDF) + TMDB poster lookup.

Design notes
------------
* Recommendations come from YOUR trained model (tfidf_matrix.pkl), ranked with a
  small genre / popularity boost. Everything is vectorised (~20 ms per query).
* TMDB is used ONLY for poster images. One search call per movie, run in
  parallel, cached on disk (.poster_cache.json) so each poster is fetched once.
* Network failures are never cached for long, so a temporary hiccup can't
  leave a poster missing forever.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

ARTIFACTS = {
    "df": BASE_DIR / "df.pkl",
    "indices": BASE_DIR / "indices.pkl",
    "tfidf": BASE_DIR / "tfidf.pkl",
    "tfidf_matrix": BASE_DIR / "tfidf_matrix.pkl",
}


def _env(name: str) -> str:
    return os.getenv(name, "").strip().strip('"').strip("'")


TMDB_API_KEY = _env("TMDB_API_KEY")
TMDB_READ_ACCESS_TOKEN = _env("TMDB_READ_ACCESS_TOKEN")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"
POSTER_CACHE_FILE = BASE_DIR / ".poster_cache.json"


# --------------------------------------------------------------------------- #
# TMDB poster lookup
# --------------------------------------------------------------------------- #
def tmdb_configured() -> bool:
    return bool(TMDB_API_KEY or TMDB_READ_ACCESS_TOKEN)


_client = httpx.Client(
    timeout=httpx.Timeout(8.0, connect=5.0),
    headers={"Authorization": f"Bearer {TMDB_READ_ACCESS_TOKEN}"} if TMDB_READ_ACCESS_TOKEN else {},
    limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
    follow_redirects=True,
)

_lock = threading.Lock()
_last_error: str | None = None
_poster_cache: dict[str, str] = {}      # casefolded title -> poster path (persisted)
_miss_until: dict[str, float] = {}      # casefolded title -> retry-after timestamp (memory only)


def _load_cache() -> None:
    try:
        data = json.loads(POSTER_CACHE_FILE.read_text())
        if isinstance(data, dict):
            _poster_cache.update({str(k): str(v) for k, v in data.items() if v})
    except (OSError, ValueError):
        pass


def _save_cache() -> None:
    try:
        tmp = POSTER_CACHE_FILE.with_suffix(".tmp")
        with _lock:
            tmp.write_text(json.dumps(_poster_cache))
        tmp.replace(POSTER_CACHE_FILE)
    except OSError:
        pass


_load_cache()


def tmdb_status() -> dict[str, Any]:
    return {"configured": tmdb_configured(), "last_error": _last_error, "cached_posters": len(_poster_cache)}


def _search(query: str) -> list[dict[str, Any]] | None:
    """Return TMDB search results, or None if the request failed (network/auth)."""
    global _last_error
    params: dict[str, Any] = {"query": query, "include_adult": "false", "language": "en-US", "page": 1}
    if TMDB_API_KEY:
        params["api_key"] = TMDB_API_KEY

    for attempt in range(3):
        try:
            resp = _client.get(TMDB_SEARCH_URL, params=params)
            if resp.status_code == 429:  # rate limited -> wait and retry
                time.sleep(min(float(resp.headers.get("Retry-After", 1)), 3))
                continue
            if resp.status_code in (401, 403):
                _last_error = "TMDB rejected the credentials (401/403). Check TMDB_API_KEY / TMDB_READ_ACCESS_TOKEN in .env"
                return None
            resp.raise_for_status()
            _last_error = None
            results = resp.json().get("results", [])
            return results if isinstance(results, list) else []
        except (httpx.HTTPError, ValueError) as exc:
            _last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.4 * (attempt + 1))
    return None


def _query_variants(title: str) -> list[str]:
    base = re.sub(r"\s+", " ", (title or "").strip())
    variants = [base, re.sub(r"\s*\(\d{4}\)$", "", base), base.split(":", 1)[0]]
    seen: list[str] = []
    for v in (x.strip() for x in variants):
        if v and v not in seen:
            seen.append(v)
    return seen


def _pick_poster(results: list[dict[str, Any]], wanted: str) -> str | None:
    with_poster = [r for r in results if r.get("poster_path")]
    if not with_poster:
        return None
    exact = [r for r in with_poster if str(r.get("title", "")).casefold() == wanted.casefold()
             or str(r.get("original_title", "")).casefold() == wanted.casefold()]
    if exact:  # several movies share a title -> take the best-known one
        return max(exact, key=lambda r: r.get("vote_count") or 0)["poster_path"]
    return with_poster[0]["poster_path"]


def poster_for(title: str) -> str | None:
    """Full poster URL for a title, or None (never raises)."""
    title = (title or "").strip()
    if not title or not tmdb_configured():
        return None
    key = title.casefold()
    with _lock:
        path = _poster_cache.get(key)
        blocked = _miss_until.get(key, 0) > time.time()
    if path:
        return f"{TMDB_IMAGE_BASE_URL}{path}"
    if blocked:
        return None

    request_failed = False
    for query in _query_variants(title):
        results = _search(query)
        if results is None:
            request_failed = True
            continue
        path = _pick_poster(results, query)
        if path:
            with _lock:
                _poster_cache[key] = path
            _save_cache()
            return f"{TMDB_IMAGE_BASE_URL}{path}"

    # Network problem -> retry soon. Genuinely no poster on TMDB -> retry in an hour.
    with _lock:
        _miss_until[key] = time.time() + (15 if request_failed else 3600)
    return None


def posters_for(titles: list[str]) -> dict[str, str | None]:
    """Look up many posters in parallel."""
    unique = list(dict.fromkeys(t for t in titles if t))
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(unique, pool.map(poster_for, unique)))


def attach_posters(movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    urls = posters_for([str(m.get("title", "")) for m in movies])
    for movie in movies:
        movie["poster_url"] = urls.get(str(movie.get("title", "")))
    return movies


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
class MovieRecommender:
    def __init__(self, df: pd.DataFrame, indices: Any, tfidf: Any, tfidf_matrix: Any) -> None:
        self.df = df.reset_index(drop=True)
        if tfidf_matrix.shape[0] != len(self.df):
            raise ValueError("tfidf_matrix rows do not match df rows - re-export both from the same notebook run")
        self.tfidf = tfidf
        # TfidfVectorizer L2-normalises rows, so a plain dot product == cosine similarity.
        self.matrix = tfidf_matrix.tocsr()

        self.titles = self.df["title"].astype(str).str.strip().to_numpy(dtype=object)
        self.title_keys = np.array([t.casefold() for t in self.titles], dtype=object)
        # `popularity` was saved as text in the pickle -> convert to numbers.
        self.popularity = pd.to_numeric(self.df["popularity"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
        self.vote = pd.to_numeric(self.df["vote_average"], errors="coerce").fillna(0).to_numpy(float)
        self.pop_score = np.log1p(self.popularity) / max(float(np.log1p(self.popularity.max())), 1.0)

        genres = self.df["genres"].fillna("").astype(str)
        try:
            vec = CountVectorizer(binary=True, token_pattern=r"[A-Za-z]+")
            self.genre_matrix = vec.fit_transform(genres).tocsr().astype(np.float32)
        except ValueError:  # no genres at all
            self.genre_matrix = self.matrix[:, :1] * 0

        # Catalog = one row per title (the most popular one), most popular first.
        self._by_title: dict[str, int] = {}
        for pos in np.argsort(-self.popularity, kind="stable"):
            self._by_title.setdefault(self.title_keys[pos], int(pos))
        self.catalog = np.fromiter(self._by_title.values(), dtype=int)

    # ---- helpers ---------------------------------------------------------- #
    def _resolve_index(self, title: str) -> int | None:
        key = (title or "").strip().casefold()
        return self._by_title.get(key) if key else None

    def _payload(self, pos: int, similarity: float | None = None) -> dict[str, Any]:
        row = self.df.iloc[pos]
        payload = {
            "title": self.titles[pos],
            "overview": row.get("overview") or "No overview available.",
            "genres": row.get("genres") or "Unknown",
            "tagline": (row.get("tagline") or "").strip(),
            "vote_average": float(self.vote[pos]),
            "popularity": float(self.popularity[pos]),
        }
        if similarity is not None:
            payload["similarity"] = float(similarity)
        return payload

    # ---- public API ------------------------------------------------------- #
    def search_titles(self, query: str, limit: int = 20) -> list[str]:
        q = (query or "").strip().casefold()
        if not q:
            return [self.titles[p] for p in self.catalog[:limit]]
        starts, contains = [], []
        for pos in self.catalog:  # popularity order -> best-known matches first
            key = self.title_keys[pos]
            if key.startswith(q):
                starts.append(self.titles[pos])
            elif q in key:
                contains.append(self.titles[pos])
            if len(starts) >= limit:
                break
        return (starts + contains)[:limit]

    def recommend(self, title: str, limit: int = 10) -> list[dict[str, Any]]:
        idx = self._resolve_index(title)
        if idx is None:
            return []

        similarity = (self.matrix @ self.matrix[idx].T).toarray().ravel()
        source = self.genre_matrix[idx]
        overlap = (self.genre_matrix @ source.T).toarray().ravel() / max(float(source.sum()), 1.0)
        # TF-IDF finds similar text; genre overlap + a small popularity nudge make the
        # list feel like real recommendations instead of keyword matches.
        score = similarity + 0.35 * overlap + 0.15 * self.pop_score
        score[self.title_keys == self.title_keys[idx]] = -np.inf  # never recommend itself

        k = min(len(score), limit * 8 + 20)
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.argsort(-score[top])]

        results, seen = [], set()
        for pos in top:
            key = self.title_keys[pos]
            if key in seen or not np.isfinite(score[pos]):
                continue
            seen.add(key)
            results.append(self._payload(int(pos), similarity[pos]))
            if len(results) >= limit:
                break
        return results

    def get_movie(self, title: str, include_images: bool = False) -> dict[str, Any] | None:
        idx = self._resolve_index(title)
        if idx is None:
            return None
        movie = self._payload(idx)
        if include_images:
            movie["poster_url"] = poster_for(movie["title"])
        return movie

    def popular_movies(self, limit: int = 12) -> list[dict[str, Any]]:
        return [self._payload(int(p)) for p in self.catalog[:limit]]

    def featured_movies(self, limit: int = 12) -> list[dict[str, Any]]:
        # Rated 10/10 by one voter is noise, so rank only well-known films by rating.
        pool = self.catalog[:3000]
        order = np.lexsort((-self.popularity[pool], -self.vote[pool]))
        return [self._payload(int(p)) for p in pool[order][:limit]]


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    loaded = {}
    for name, path in ARTIFACTS.items():
        with open(path, "rb") as fh:
            loaded[name] = pickle.load(fh)
    df = loaded["df"]
    if "title" not in df.columns:
        raise ValueError("df.pkl is missing the required title column")
    return MovieRecommender(df, loaded["indices"], loaded["tfidf"], loaded["tfidf_matrix"])


if __name__ == "__main__":
    # Quick self-test:  python recommender.py
    print("TMDB configured:", tmdb_configured())
    rec = get_recommender()
    print("Movies loaded:", len(rec.catalog))
    print("Recommendations for Toy Story:", [m["title"] for m in rec.recommend("Toy Story", 5)])
    print("Poster URL:", poster_for("Toy Story"))
    print("TMDB status:", tmdb_status())