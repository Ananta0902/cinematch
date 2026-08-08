# 🎬 CineMatch

### ML-Powered Content-Based Movie Recommendation System

CineMatch is a **machine learning-based movie recommendation system** that recommends movies based on their content, including genres, keywords, and textual movie information.

The recommendation engine uses **Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity** to identify movies with similar content.

The ML model is served through a **FastAPI backend** and integrated with a **Streamlit frontend**.

---

## 🚀 Live Demo

🎬 **Frontend:**  
https://cinemuvicorn-frontend.onrender.com/

⚙️ **Backend API:**  
https://cinemuvicorn-api.onrender.com

📚 **API Documentation:**  
https://cinemuvicorn-api.onrender.com/docs

---

## 🧠 Machine Learning Approach

CineMatch follows a **Content-Based Filtering** approach.

### Recommendation Pipeline

```text
Movie Dataset
      ↓
Data Preprocessing
      ↓
Combine Movie Features
      ↓
TF-IDF Vectorization
      ↓
TF-IDF Feature Matrix
      ↓
Cosine Similarity
      ↓
Rank Similar Movies
      ↓
Top-K Recommendations
```

How it works
1.Movie information is processed and converted into textual features.
2.TF-IDF converts the text into numerical feature vectors.
3.Cosine similarity measures the similarity between movie vectors.
4.Movies are ranked based on similarity scores.
5.The top similar movies are returned as recommendations.

CineMatch also supports natural-language descriptions, allowing users to describe the type of movie they want and receive relevant recommendations.

✨ Features
🎬 Content-based movie recommendations
🧠 NLP-based recommendation engine
🔤 TF-IDF vectorization
📐 Cosine similarity
🔍 Movie search
🤖 Description-based recommendations
🎭 Genre filtering
⭐ IMDb rating filtering
📅 Release-year filtering
🎞️ Movie details and posters
💡 Similarity-based recommendation explanations

🛠️ Tech Stack
**Machine Learning**
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Cosine Similarity
**Backend**
FastAPI
Uvicorn
Pydantic
HTTPX
**Frontend**
Streamlit
**External API**
OMDb API
**Deployment**
Render
GitHub

📂 Project Structure
CineMatch/
│
├── app.py
├── main.py
├── df.pkl
├── indices.pkl
├── tfidf.pkl
├── tfidf_matrix.pkl
├── requirements.txt
├── README.md
│
└── assets/
    └── no_poster.png

⚙️ Run Locally
Clone the repository
git clone https://github.com/Ananta0902/cinematch.git
cd cinematch
Create virtual environment
python -m venv .venv
Activate
.venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Add environment variable
Create a .env file:
OMDB_API_KEY=your_api_key
Start FastAPI
uvicorn main:app --reload
Start Streamlit
Open another terminal:
streamlit run app.py

🔮 Future Improvements
Hybrid recommendation system
Transformer-based embeddings
User personalization
Recommendation evaluation using Precision@K / Recall@K
Recommendation caching
CI/CD pipeline
