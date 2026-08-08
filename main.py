import os
import asyncio
import pickle
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

#env
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_BASE = "https://www.omdbapi.com/"
if not OMDB_API_KEY:
    raise RuntimeError(
        "OMDB_API_KEY missing. Add it to .env"
    )

# FASTAPI APP
app = FastAPI(title="Movie Recommender API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],    #allow all http method get,put,post,delete
    allow_headers=["*"],
)

# PICKLE GLOBALS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
HOME_CACHE = None
MOVIE_CARD_CACHE = {}
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# MODELS
class MovieCard(BaseModel):
    imdb_id: str
    title: str
    year: Optional[str] = None
    poster_url: Optional[str] = None
    movie_type: Optional[str] = None
    genre: Optional[str] = None
    imdb_rating: Optional[str] = None
    common_tags: List[str] = []

class MovieDetails(BaseModel):
    imdb_id: str
    title: str
    year: Optional[str] = None
    plot: Optional[str] = None
    poster_url: Optional[str] = None
    genre: Optional[str] = None
    director: Optional[str] = None
    actors: Optional[str] = None
    runtime: Optional[str] = None
    imdb_rating: Optional[str] = None

class RecommendationItem(BaseModel):
    title: str
    score: float
    movie: Optional[MovieCard] = None
    reason: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    movie_details: MovieDetails
    recommendations: List[RecommendationItem]

# UTILS
def _norm_title(title: str) -> str:
    return str(title).strip().lower()

async def omdb_get(params: Dict[str, Any]) -> Dict[str, Any]:

    params = dict(params)
    params["apikey"] = OMDB_API_KEY

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(8.0, connect=3.0)
        ) as client:

            response = await client.get(
                OMDB_BASE,
                params=params
            )

    except httpx.RequestError as e:

        raise HTTPException(
            status_code=502,
            detail="OMDb service temporarily unavailable."
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=502,
            detail="Unable to contact OMDb API."
        )

    try:
        data = response.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Invalid response from OMDb API."
        )

    if data.get("Response") == "False":

        raise HTTPException(
            status_code=404,
            detail=data.get("Error", "Movie not found")
        )

    return data

async def attach_movie_card(title: str):

    try:
        movie = await search_first(title)

        if not movie:
            return None

        details = await get_movie_details(movie["imdbID"])

        return {
            "imdb_id": details.imdb_id,
            "title": details.title,
            "poster_url": details.poster_url,
            "year": details.year,
            "genre": details.genre,
            "imdb_rating": details.imdb_rating,
        }

    except HTTPException:
        return None

    except Exception:
        return None

async def search_movies(query: str) -> List[dict]:
    """
    Search movies by title.
    """

    data = await omdb_get(
        {
            "s": query
        }
    )

    return data.get("Search", [])

async def search_first(query: str) -> Optional[dict]:

    movies = await search_movies(query)

    if len(movies) == 0:
        return None

    return movies[0]

async def get_movie_details(imdb_id: str) -> MovieDetails:

    data = await omdb_get(
        {
            "i": imdb_id,
            "plot": "full"
        }
    )

    return MovieDetails(
        imdb_id=data.get("imdbID", ""),
        title=data.get("Title", ""),
        year=data.get("Year"),
        plot=data.get("Plot"),
        poster_url=data.get("Poster"),
        genre=data.get("Genre"),
        director=data.get("Director"),
        actors=data.get("Actors"),
        runtime=data.get("Runtime"),
        imdb_rating=data.get("imdbRating")
    )

def movie_card_from_search(movie: dict) -> MovieCard:

    return MovieCard(
        imdb_id=movie.get("imdbID", ""),
        title=movie.get("Title", ""),
        year=movie.get("Year"),
        poster_url=movie.get("Poster"),
        movie_type=movie.get("Type")
    )
#TF-IDF Helpers
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )

def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )

def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10
) -> List[dict]:

    global df, tfidf_matrix

    if df is None or tfidf_matrix is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF resources not loaded"
        )

    idx = get_local_idx_by_title(query_title)

    query_vector = tfidf_matrix[idx]

    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()

    query_tags = set(str(df.iloc[idx]["tags"]).split())

    order = np.argsort(-scores)

    recommendations = []

    for i in order:

        if i == idx:
            continue

        row = df.iloc[int(i)]

        title = row["title"]

        movie_tags = set(str(row["tags"]).split())

        common_tags = list(query_tags & movie_tags)

        # remove useless words
        stop_words = {
            "the","a","an","and","or","of","to","in","on","for",
            "with","his","her","their","its","is","are","was",
            "be","by","from","at","as","into","about","after",
            "before","this","that","these","those", "released",
    "recently",
    "movie",
    "film",
    "story",
    "based",
    "tells",
    "life",
    "lives",
    "family",
    "young",
    "years",
    "people",
    "find",
    "must",
    "becomes",
    "become",
    "around",
    "takes",
    "named"
        }

        common_tags = [
            word
            for word in common_tags
            if len(word) > 3 and word.lower() not in stop_words
        ]

        common_tags = common_tags[:4]

        if common_tags:
            reason = " • ".join(common_tags[:3])

        else:

            reason = "Similar story themes"

        genre = str(row.get("genres", "")).replace("|", ", ")

        recommendations.append({
        "title": title,
        "score": float(scores[i]),
        "reason": reason,
        "genre": genre
        }
        )

        if len(recommendations) == top_n:
            break

    return recommendations

def explain_similarity(query_title: str, recommended_title: str) -> str:
    """
    Generate a human readable explanation of why
    two movies are considered similar.
    """

    idx1 = get_local_idx_by_title(query_title)
    idx2 = get_local_idx_by_title(recommended_title)

    tags1 = set(str(df.iloc[idx1]["tags"]).lower().split())
    tags2 = set(str(df.iloc[idx2]["tags"]).lower().split())

    common = list(tags1 & tags2)

    ignore = {
        "movie","film","story","one","man","woman","life",
        "young","new","find","must","world","day","girl",
        "boy","get","go","take","make","time","years"
    }

    common = [w for w in common if len(w) > 3 and w not in ignore]

    common = sorted(common)[:4]

    if not common:
        return "Similar themes and story style."

    return "Both movies feature " + ", ".join(common) + "."


def description_search(description: str, top_n: int = 5):

    global df, tfidf_obj, tfidf_matrix

    query_vector = tfidf_obj.transform([description])
    scores = (query_vector @ tfidf_matrix.T).toarray().ravel()

    order = np.argsort(-scores)

    results = []

    for idx in order:

        if scores[idx] <= 0:
            continue

        results.append({
            "title": df.iloc[idx]["title"],
            "score": round(float(scores[idx]), 4)
        })

        if len(results) == top_n:
            break

    return results

# STARTUP: LOAD PICKLES
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix (usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

# ROUTES
@app.get("/health")
def health():
    return {
    "status": "healthy",
    "model_loaded": tfidf_matrix is not None,
    "movies_loaded": df is not None
}
#home route
@app.get("/home", response_model=List[MovieCard])
async def home(limit: int = Query(8, ge=1, le=20)):

    if df is None:
        raise HTTPException(
            status_code=500,
            detail="Dataset not loaded"
        )

    featured = (
        df.dropna(subset=["vote_average"])
        .sort_values(
            by="vote_average",
            ascending=False
        )
        .head(limit)
    )

    async def get_card(title):

        try:
            return await attach_movie_card(title)

        except Exception as e:

            

            return None

    results = await asyncio.gather(
        *(get_card(title) for title in featured["title"].tolist())
    )

    cards = [
        card
        for card in results
        if card is not None
    ]

    return cards

#search route
@app.get("/search", response_model=List[MovieCard])
async def search_route(
    query: str = Query(..., min_length=1)
):

    movies = await search_movies(query)

    async def build_card(movie):

        try:

            details = await get_movie_details(
                movie["imdbID"]
            )

            return MovieCard(
                imdb_id=details.imdb_id,
                title=details.title,
                year=details.year,
                poster_url=details.poster_url,
                movie_type=movie.get("Type"),
                genre=details.genre,
                imdb_rating=details.imdb_rating
            )

        except Exception as e:

            

            return None

    results = await asyncio.gather(
        *(build_card(movie) for movie in movies[:10])
    )

    return [
        card
        for card in results
        if card is not None
    ]

#movie details route 
@app.get("/movie/{imdb_id}", response_model=MovieDetails)
async def movie_route(imdb_id: str):

    return await get_movie_details(imdb_id)

#recommend function
@app.get("/recommend", response_model=SearchResponse)
async def recommend(query: str):

    movie = await search_first(query)

    if movie is None:
        raise HTTPException(
            status_code=404,
            detail="Movie not found"
        )

    details = await get_movie_details(movie["imdbID"])

    tfidf_titles = tfidf_recommend_titles(
        details.title,
        top_n=8
    )

    async def get_card(item):

        try:
            card = await attach_movie_card(item["title"])

            if card is None:
                return None

            return RecommendationItem(
                title=item["title"],
                score=round(item["score"] * 100, 1),
                movie=card,
                reason=item["reason"]
            )

        except Exception as e:

            

            return None

    results = await asyncio.gather(
        *(get_card(item) for item in tfidf_titles)
    )

    recommendations = [
        result
        for result in results
        if result is not None
    ]

    return SearchResponse(
        query=query,
        movie_details=details,
        recommendations=recommendations
    )

@app.get("/recommend/description")
async def recommend_description(
    description: str,
    top_n: int = 5
):

    matches = description_search(description, top_n)

    results = await asyncio.gather(
        *(
            attach_movie_card(item["title"])
            for item in matches
        ),
        return_exceptions=True
    )

    recommendations = []

    for item, card in zip(matches, results):

        if isinstance(card, dict):

            recommendations.append(
                {
                    "title": item["title"],
                    "score": item["score"],
                    "movie": card
                }
            )

    return {
        "recommendations": recommendations
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )