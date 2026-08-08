import os
import requests
import streamlit as st

# CONFIG
API_BASE = "https://cinemuvicorn-api.onrender.com"

NO_POSTER = os.path.join("assets", "no_poster.png")

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# CSS
st.markdown("""
<style>

.block-container{
    max-width:1400px;
    padding-top:1rem;
}

.movie-title{
    font-size:18px;
    font-weight:700;
    text-align:center;
    min-height:55px;
    margin-top:8px;
}

.small-text{
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# SESSION
if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

if "view" not in st.session_state:
    st.session_state.view = "home"

if "selected_imdb_id" not in st.session_state:
    st.session_state.selected_imdb_id = None

if "home_movies" not in st.session_state:
    st.session_state.home_movies = None

qp = st.query_params

if qp.get("view") == "details":
    st.session_state.view = "details"

if qp.get("id"):
    st.session_state.selected_imdb_id = qp.get("id")

# NAVIGATION
def goto_home():

    st.session_state.view = "home"

    st.query_params["view"] = "home"

    if "id" in st.query_params:
        del st.query_params["id"]

    st.rerun()


def goto_details(imdb_id):

    st.session_state.view = "details"

    st.session_state.selected_imdb_id = imdb_id

    st.query_params["view"] = "details"

    st.query_params["id"] = imdb_id

    st.rerun()

# API
@st.cache_data(ttl=60)
def api_get_json(path, params=None):

    try:

        r = requests.get(
            f"{API_BASE}{path}",
            params=params,
            timeout=45
        )

        if r.status_code != 200:

            try:
                error_data = r.json()

                error_message = error_data.get(
                    "detail",
                    f"Server error: {r.status_code}"
                )

            except Exception:

                error_message = (
                    f"Server returned HTTP {r.status_code}"
                )

            return None, error_message

        try:

            return r.json(), None

        except Exception:

            return None, "Server returned invalid JSON."

    except requests.exceptions.Timeout:

        return None, (
            "The recommendation server took too long to respond. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:

        return None, (
            "Cannot connect to the recommendation server. "
            "Make sure FastAPI/Uvicorn is running."
        )

    except Exception as e:

        return None, str(e)


@st.cache_data(ttl=3600)
def valid_image(url):

    if not url:
        return False

    if url == "N/A":
        return False

    try:
        r = requests.get(
            url,
            stream=True,
            timeout=5
        )

        if r.status_code != 200:
            return False

        content = r.headers.get("content-type", "")

        return "image" in content.lower()

    except:
        return False


def show_poster(url):

    if valid_image(url):
        st.image(url, use_container_width=True)
    else:
        st.image(NO_POSTER, use_container_width=True)

def recommendation_cards(recommendations):

    cards = []

    for item in recommendations:

        movie = item.get("movie")

        if movie:

            m = movie.copy()

            m["score"] = item.get("score")
            m["reason"] = item.get("reason")
            m["genre"] = movie.get("genre")

            cards.append(m)

    return cards

def apply_filters(cards):

    if not cards:
        return []

    filtered = []

    for movie in cards:

        # GENRE
        if genre_filter != "All":

            genre = str(
                movie.get("genre") or ""
            )

            if genre_filter.lower() not in genre.lower():
                continue

        # RATING
        try:

            rating_value = movie.get("imdb_rating")

            if rating_value in [None, "", "N/A"]:
                continue

            rating = float(rating_value)

            if rating < rating_filter:
                continue

        except (ValueError, TypeError):

            continue

        # YEAR
        try:

            year_value = movie.get("year")

            if year_value in [None, "", "N/A"]:
                continue

            year = int(
                str(year_value)[:4]
            )

            if year < year_filter:
                continue

        except (ValueError, TypeError):

            continue

        filtered.append(movie)

    return filtered

def poster_grid(cards, cols=4, key_prefix="grid"):

    if not cards:

        st.info("No movies found.")

        return

    rows = (len(cards) + cols - 1) // cols

    index = 0

    for r in range(rows):

        columns = st.columns(cols)

        for c in range(cols):

            if index >= len(cards):
                break

            movie = cards[index]

            index += 1

            imdb_id = movie.get("imdb_id")

            title = movie.get("title", "Unknown")

            poster = movie.get("poster_url")

            score = movie.get("score")
            reason = movie.get("reason")
            genre = movie.get("genre")
            common_tags = movie.get("common_tags", [])

            with columns[c]:

                show_poster(poster)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>",
                    unsafe_allow_html=True
                )

                if score is not None:
                    if score >= 90:
                        badge = "🔥 Excellent Match"
                    elif score >= 75:
                        badge = "⭐ Great Match"
                    elif score >= 60:
                        badge = "👍 Good Match"
                    else:
                        badge = "🎬 Similar"

                    st.caption(f"{badge} • {score:.1f}% Match")
                if genre:
                    st.caption(f"🎭Genre: {genre}")
                if reason:
                    st.caption(f"💡 {reason}")
                
                if st.button(
                    "🎬 View Details",
                    key=f"{key_prefix}_{index}"
                ):

                    goto_details(imdb_id)
                
# SIDEBAR
with st.sidebar:

    st.title("🎬 Movie Recommender")

    if st.button("🏠 Home"):

        goto_home()
    st.divider()

    grid_cols = st.slider(

        "Grid Columns",

        3,

        6,

        4

    )
    st.divider()

    genre_filter = st.selectbox(
    "🎭 Genre",
    [
        "All",
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Drama",
        "Fantasy",
        "Horror",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller"
    ]
    )

    rating_filter = st.slider(
    "⭐ Minimum IMDb Rating",
    0.0,
    10.0,
    0.0,
    0.5
    )

    year_filter = st.slider(
    "📅 Release Year",
    1950,
    2025,
    1950
    )

# TITLE

st.title("🎬 Movie Recommendation System")

search = st.text_input(
    "🔍 Search Movie",
    placeholder="Interstellar, Batman..."
)

description = st.text_area(
    "🤖 Describe the movie you want to watch",
    placeholder="A comedy movie with lots of funny moments...",
    height=90,
)

recommend_btn = st.button("✨ Recommend by Description")

st.divider()

# ---------------- HOME PAGE ----------------

if st.session_state.view == "home":

    # ---------------- SEARCH ---------------- #

    if search.strip():

        movies, err = api_get_json(
            "/search",
            {"query": search}
        )

        if not err:
            st.session_state.search_results = movies

    if st.session_state.search_results:

        st.subheader("🔍 Search Results")

        st.caption(
            f"Filters: "
            f"Genre = {genre_filter if genre_filter != 'All' else 'Any'} • "
            f"IMDb ≥ {rating_filter} • "
            f"Year ≥ {year_filter}"
        )

        cards = apply_filters(
            st.session_state.search_results
        )

        poster_grid(
            cards,
            cols=grid_cols,
            key_prefix="search",
        )

    st.divider()

    # ---------------- DESCRIPTION SEARCH ---------------- #

    if recommend_btn:

        if not description.strip():

            st.warning("Please enter a movie description.")

        else:

            with st.spinner("Finding movies..."):

                bundle, err = api_get_json(
                    "/recommend/description",
                    {
                        "description": description,
                        "top_n": 6,
                    },
                )

            if err:

                st.error(err)

            else:

                st.session_state.ai_results = bundle

    # ---------- ALWAYS SHOW PREVIOUS AI RESULTS ---------- #

    if st.session_state.ai_results:

        recommendations = st.session_state.ai_results.get(
            "recommendations",
            []
        )

        st.subheader("🤖 AI Recommendations")

        st.caption(
            f"Filters: "
            f"Genre = {genre_filter if genre_filter != 'All' else 'Any'} • "
            f"IMDb ≥ {rating_filter} • "
            f"Year ≥ {year_filter}"
        )

        if recommendations:

            cards = recommendation_cards(
                recommendations
            )

            cards = apply_filters(cards)

            poster_grid(
                cards,
                cols=grid_cols,
                key_prefix="description",
            )

        else:

            st.warning("No recommendations found.")

        st.divider()

    # ---------------- FEATURED MOVIES ---------------- #

    st.subheader("🔥 Featured Movies")

    if st.session_state.home_movies is None:

        home_movies, home_err = api_get_json(
            "/home"
        )

        if home_err:

            st.error(home_err)

        else:

            st.session_state.home_movies = home_movies

    if st.session_state.home_movies:

        featured_cards = apply_filters(
            st.session_state.home_movies
        )

        poster_grid(
            featured_cards,
            cols=grid_cols,
            key_prefix="home"
        )

    else:

        st.warning("No featured movies available.")


# ---------------- DETAILS PAGE ----------------

elif st.session_state.view == "details":

    imdb_id = st.session_state.selected_imdb_id

    if not imdb_id:

        st.warning("No movie selected.")

        st.stop()

    movie, err = api_get_json(
        f"/movie/{imdb_id}"
    )

    if err:

        st.error(err)

        st.stop()

    if not movie:

        st.error("Movie not found.")

        st.stop()

    # ---------------- HEADER ---------------- #

    c1, c2 = st.columns([1, 5])

    with c1:

        if st.button("⬅ Back"):

            goto_home()

    with c2:

        st.title(
            movie.get("title", "")
        )

    st.divider()

    # ---------------- MOVIE INFO ---------------- #

    left, right = st.columns(
        [1, 2],
        gap="large"
    )

    with left:

        show_poster(
            movie.get("poster_url")
        )

        st.markdown(
            f"### ⭐ {movie.get('imdb_rating', 'N/A')}"
        )

        st.markdown(
            f"**🎭 Genre**  \n"
            f"{movie.get('genre', 'N/A')}"
        )

        st.markdown(
            f"**🎬 Director**  \n"
            f"{movie.get('director', 'N/A')}"
        )

        st.markdown(
            f"**⏱ Runtime**  \n"
            f"{movie.get('runtime', 'N/A')}"
        )

    with right:

        st.subheader("Actors")

        st.write(
            movie.get("actors", "N/A")
        )

        st.write("")

        st.subheader("Year")

        st.write(
            movie.get("year", "N/A")
        )

        st.write("")

        st.subheader("Overview")

        plot = movie.get("plot")

        if plot and plot != "N/A":

            st.write(plot)

        else:

            st.info(
                "No overview available."
            )

    st.divider()

    # ---------------- RECOMMENDATIONS ---------------- #

    st.subheader("🎯 Similar Movies")

    with st.spinner(
        "Finding similar movies..."
    ):

        bundle, err = api_get_json(
            "/recommend",
            {
                "query": movie["title"]
            }
        )

    if err:

        st.warning(
            """
Movies similar to this movie are not available
in our recommendation dataset.

Try popular movies like:

• Interstellar

• Inception

• Batman Begins

• Avengers

• The Dark Knight
"""
        )

    else:

        cards = recommendation_cards(
            bundle.get(
                "recommendations",
                []
            )
        )

        cards = apply_filters(cards)

        poster_grid(
            cards,
            cols=grid_cols,
            key_prefix="recommend"
        )
    