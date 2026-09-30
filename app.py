from __future__ import annotations

import html
from typing import Any
from urllib.parse import quote

import streamlit as st

from recommender import attach_posters, get_recommender, tmdb_configured, tmdb_status

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { --panel:#121b33; --border:rgba(255,255,255,.10); --text:#f1f5ff; --muted:#9ca9c1; }
        html, body, [class*="css"] { font-family:Manrope,sans-serif; }
        .stApp { background:radial-gradient(circle at 10% 0%,#1c3155 0%,transparent 28%),linear-gradient(180deg,#090f1f,#0d1427); color:var(--text); }
        .block-container { max-width:1280px; padding-top:4.5rem; padding-bottom:3rem; }
        header[data-testid="stHeader"] { background:transparent; }
        [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none!important; visibility:hidden; }
        .brand { font-size:2.4rem; font-weight:800; margin:0; letter-spacing:-.07rem; }
        .subhead { color:var(--muted); margin:.35rem 0 1.5rem; }
        .section-title { font-size:1.28rem; font-weight:800; margin:1.7rem 0 .7rem; }
        .poster-card { display:block; overflow:hidden; background:var(--panel); border:1px solid var(--border); border-radius:16px; color:var(--text)!important; text-decoration:none!important; box-shadow:0 10px 28px rgba(0,0,0,.2); transition:transform .18s ease,border-color .18s ease; margin-bottom:.4rem; }
        .poster-card:hover { transform:translateY(-5px); border-color:rgba(255,189,74,.85); }
        .poster-frame { position:relative; display:block; width:100%; aspect-ratio:2/3; background:#17213a; overflow:hidden; }
        .poster-frame img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }
        .poster-fallback { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.4rem; padding:.8rem; text-align:center; color:#9ca9c1; font-size:.85rem; font-weight:700; }
        .poster-fallback b { font-size:2rem; }
        .poster-card .movie-name { display:block; overflow:hidden; padding:.7rem .75rem .15rem; font-size:.92rem; font-weight:800; white-space:nowrap; text-overflow:ellipsis; }
        .poster-card .movie-meta { display:block; padding:0 .75rem .75rem; color:var(--muted); font-size:.78rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .detail-poster { border-radius:18px; box-shadow:0 14px 34px rgba(0,0,0,.35); }
        .detail-panel { padding:1.5rem; border:1px solid var(--border); border-radius:22px; background:linear-gradient(135deg,rgba(25,37,67,.96),rgba(15,23,43,.94)); }
        .detail-title { margin:0 0 .4rem; font-size:2.1rem; line-height:1.1; }
        .detail-meta { color:#b7c3dc; margin:.4rem 0 1rem; }
        .detail-copy { color:#d8e0f1; line-height:1.65; max-width:55rem; }
        .genre { display:inline-block; margin-right:.4rem; padding:.3rem .6rem; border-radius:99px; color:#d1f4ff; background:rgba(90,207,255,.12); border:1px solid rgba(90,207,255,.22); font-size:.78rem; }
        .search-panel { overflow:hidden; margin-top:-.7rem; border:1px solid rgba(255,255,255,.13); border-radius:0 0 14px 14px; background:#141926; box-shadow:0 16px 32px rgba(0,0,0,.38); }
        .search-suggestion { display:flex; align-items:center; gap:.75rem; min-height:48px; padding:.55rem .8rem; color:#f2f5ff!important; text-decoration:none!important; font-weight:700; }
        .search-suggestion:hover { background:#242b3d; }
        .search-empty { display:flex; align-items:center; gap:.75rem; min-height:48px; padding:.55rem .8rem; color:#9ca9c1; font-weight:600; }
        .search-icon { color:#b9c4d8; font-size:1.2rem; line-height:1; }
        .search-caption { overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
        div[data-testid="stButton"] > button { border-radius:10px; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data helpers (everything runs in-process: no separate API server needed)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading recommendation model…")
def load_model():
    return get_recommender()


@st.cache_data(ttl=3600, show_spinner=False, max_entries=500)
def cached_search(query: str) -> list[str]:
    return load_model().search_titles(query, 10)


def short_text(value: Any, length: int = 220) -> str:
    text = " ".join(str(value or "No description available.").split())
    return text if len(text) <= length else f"{text[:length].rsplit(' ', 1)[0]}…"


# Only the (fast, deterministic) model results are cached here. Posters are attached
# OUTSIDE the cache: recommender.poster_for() has its own disk cache, and a temporary
# TMDB failure must not get frozen into a cached page.
@st.cache_data(ttl=3600, show_spinner=False)
def browse_movies(kind: str, limit: int) -> list[dict[str, Any]]:
    model = load_model()
    return model.popular_movies(limit) if kind == "popular" else model.featured_movies(limit)


@st.cache_data(ttl=3600, show_spinner=False)
def movie_details(title: str, limit: int) -> dict[str, Any]:
    model = load_model()
    movie = model.get_movie(title)
    if movie is None:
        return {"found": False}
    return {"found": True, "movie": movie, "recommendations": model.recommend(title, limit)}


def browse(kind: str, limit: int) -> list[dict[str, Any]]:
    with st.spinner("Loading posters…"):
        return attach_posters(browse_movies(kind, limit))


def details(title: str, limit: int) -> dict[str, Any]:
    payload = movie_details(title, limit)
    if payload.get("found"):
        with st.spinner("Loading posters…"):
            attach_posters([payload["movie"]])
            attach_posters(payload["recommendations"])
    return payload


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def poster_frame(movie: dict[str, Any]) -> str:
    """Poster image with a title/emoji fallback underneath (visible if the image is missing)."""
    title = html.escape(str(movie.get("title") or "Untitled"))
    url = movie.get("poster_url")
    img = f'<img src="{html.escape(str(url), quote=True)}" alt="" loading="lazy" referrerpolicy="no-referrer">' if url else ""
    return f'<span class="poster-frame"><span class="poster-fallback"><b>🎬</b>{title}</span>{img}</span>'


def render_poster_card(movie: dict[str, Any]) -> str:
    title_raw = str(movie.get("title") or "Untitled")
    title = html.escape(title_raw)
    genres = html.escape(short_text(movie.get("genres") or "Movie", 28))
    rating = float(movie.get("vote_average") or 0)
    return (
        f'<a class="poster-card" href="?movie={quote(title_raw)}" target="_self" aria-label="Open {title}">'
        f'{poster_frame(movie)}'
        f'<span class="movie-name">{title}</span><span class="movie-meta">⭐ {rating:.1f} · {genres}</span></a>'
    )


def render_grid(movies: list[dict[str, Any]], columns: int = 6) -> None:
    for start in range(0, len(movies), columns):
        chunk = movies[start : start + columns]
        for column, movie in zip(st.columns(columns), chunk):
            with column:
                st.markdown(render_poster_card(movie), unsafe_allow_html=True)


def render_search_suggestions(titles: list[str], query: str = "") -> str:
    if not titles:
        return (
            '<div class="search-panel"><div class="search-empty">'
            f'<span class="search-icon">⌕</span><span>No results found for “{html.escape(query)}”</span></div></div>'
        )
    rows = []
    for title_raw in titles[:10]:
        title = html.escape(str(title_raw))
        url = html.escape(quote(str(title_raw)), quote=True)
        rows.append(
            f'<a class="search-suggestion" href="?movie={url}" target="_self">'
            f'<span class="search-icon">⌕</span><span class="search-caption">{title}</span></a>'
        )
    return f'<div class="search-panel">{"".join(rows)}</div>'


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
inject_styles()
model = load_model()
selected_title = str(st.query_params.get("movie") or "")

if not tmdb_configured():
    st.warning(
        "No TMDB credentials found, so posters cannot be loaded. Create a `.env` file next to app.py with "
        "`TMDB_API_KEY=...` (or `TMDB_READ_ACCESS_TOKEN=...`) and restart."
    )

header_left, header_right = st.columns([4, 2])
with header_left:
    st.markdown('<h1 class="brand">CineMatch</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subhead">Pick a poster to open its story and find movies with a similar feel.</p>', unsafe_allow_html=True)
with header_right:
    search = st.text_input("Search movies", placeholder="Try Toy Story", label_visibility="collapsed")
    if len(search.strip()) >= 2:  # skip 1-letter queries: cheap on CPU, useless results
        titles = cached_search(search.strip().casefold())
        st.markdown(render_search_suggestions(titles, search.strip()), unsafe_allow_html=True)

if selected_title:
    if st.button("← Back to browse", key="back-to-browse"):
        st.query_params.clear()
        st.rerun()

    payload = details(selected_title, 12)
    if not payload.get("found"):
        st.error("That movie was not found. Choose a title from the browse sections.")
    else:
        movie = payload["movie"]
        poster_column, info_column = st.columns([1, 2.4], gap="large")
        with poster_column:
            st.markdown(f'<div class="detail-poster">{poster_frame(movie)}</div>', unsafe_allow_html=True)
        with info_column:
            title = html.escape(str(movie.get("title") or selected_title))
            genre_tags = "".join(f'<span class="genre">{html.escape(g)}</span>' for g in str(movie.get("genres") or "Movie").split())
            tagline = html.escape(short_text(movie.get("tagline"), 120)) if movie.get("tagline") else ""
            st.markdown(
                f'<div class="detail-panel"><h2 class="detail-title">{title}</h2>'
                f'<div class="detail-meta">⭐ {float(movie.get("vote_average") or 0):.1f} · Popularity {float(movie.get("popularity") or 0):.0f}</div>'
                f'{genre_tags}'
                f'{f"<p class=detail-copy><i>{tagline}</i></p>" if tagline else ""}'
                f'<p class="detail-copy">{html.escape(short_text(movie.get("overview"), 400))}</p></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="section-title">Related to this movie</div>', unsafe_allow_html=True)
        related = payload.get("recommendations") or []
        if related:
            render_grid(related, columns=6)
        else:
            st.info("No related movies are available for this title yet.")
else:
    st.markdown('<div class="section-title">Popular now</div>', unsafe_allow_html=True)
    render_grid(browse("popular", 12), columns=6)
    st.markdown('<div class="section-title">Top rated picks</div>', unsafe_allow_html=True)
    render_grid(browse("featured", 12), columns=6)

if tmdb_configured() and tmdb_status().get("last_error"):
    st.caption(f"⚠️ TMDB: {tmdb_status()['last_error']}")