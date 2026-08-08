# 🎬 CineMatch

### ML-Powered Content-Based Movie Recommendation System

CineMatch is a **machine learning-powered movie recommendation system** that recommends movies based on their content, including **genres, keywords, and textual movie information**.

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

### 🔄 Recommendation Pipeline

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

### How It Works

1. **Data Preprocessing**
   Movie information is cleaned and processed to create meaningful textual features.

2. **Feature Engineering**
   Relevant movie attributes such as genres, keywords, and descriptions are combined into a single textual representation.

3. **TF-IDF Vectorization**
   TF-IDF converts the textual movie information into numerical feature vectors.

4. **Cosine Similarity**
   Cosine similarity measures how similar two movie vectors are based on their content.

5. **Recommendation Ranking**
   Movies are ranked according to their similarity scores.

6. **Top-K Recommendations**
   The most similar movies are returned as recommendations.

CineMatch also supports **natural-language descriptions**, allowing users to describe the type of movie they are looking for and receive relevant recommendations.

---

## ✨ Features

* 🎬 Content-based movie recommendations
* 🧠 NLP-powered recommendation engine
* 🔤 TF-IDF vectorization
* 📐 Cosine similarity
* 🔍 Movie search
* 🤖 Description-based recommendations
* 🎭 Genre filtering
* ⭐ IMDb rating filtering
* 📅 Release-year filtering
* 🎞️ Movie details and posters
* 💡 Similarity-based recommendation explanations

---

## 🛠️ Tech Stack

### 🤖 Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF
* Cosine Similarity

### ⚙️ Backend

* FastAPI
* Uvicorn
* Pydantic
* HTTPX

### 🎨 Frontend

* Streamlit

### 🎬 External API

* OMDb API

### ☁️ Deployment & Tools

* Render
* GitHub

---

## 📂 Project Structure

```text
CineMatch/
│
├── app.py
├── main.py
│
├── df.pkl
├── indices.pkl
├── tfidf.pkl
├── tfidf_matrix.pkl
│
├── requirements.txt
├── README.md
│
└── assets/
    └── no_poster.png
```

---

## ⚙️ Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ananta0902/cinematch.git
cd cinematch
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

### 3️⃣ Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Add Environment Variables

Create a `.env` file in the project root:

```env
OMDB_API_KEY=your_api_key
```

Replace `your_api_key` with your actual OMDb API key.

### 6️⃣ Start the FastAPI Backend

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 7️⃣ Start the Streamlit Frontend

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🔮 Future Improvements

* 🔀 Hybrid recommendation system combining content-based and collaborative filtering
* 🤖 Transformer-based embeddings for improved semantic similarity
* 👤 User-specific recommendation personalization
* 📊 Recommendation evaluation using **Precision@K** and **Recall@K**
* ⚡ Recommendation caching for improved performance
* 🔄 CI/CD pipeline for automated testing and deployment

---
