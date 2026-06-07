# 🎬 Movie AI Recommender

An AI-powered Movie Recommendation System built with a premium dark-themed **Streamlit** dashboard. The app suggests movies based on description similarity using a **Content-Based Filtering** approach powered by **TF-IDF vector embeddings** and a **K-Nearest Neighbors (KNN)** model.

---

## ⭐️ Features

- **Content-Based Recommendation Engine:** Employs NLP techniques to analyze movie descriptions and fetch recommendations based on vector distance.
- **OMDb API Integration:** Dynamically fetches movie posters, ratings, genres, directors, and runtimes to display detailed, live information for both the selected movie and recommended suggestions.
- **Smart API Caching:** Utilizes Streamlit's `@st.cache_data` to cache OMDb API requests, ensuring instant loading times and minimizing API rate limits.
- **Interactive IMDB Links:** Posters and recommendation cards are fully interactive and link directly to their respective IMDb pages for quick navigation.

---

## 🛠 Tech Stack & ML Libraries

The project is implemented using the following technologies and packages:

| Component | Technology / Library | Role in Project |
| :--- | :--- | :--- |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Dashboard presentation and interactivity |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | Text preprocessing (TF-IDF) & similarity searching (KNN) |
| **Data Handling** | [Pandas](https://pandas.pydata.org/) | Dataset manipulation, filtering, and indexing |
| **Math / Matrices** | [NumPy](https://numpy.org/) | Matrix computations and vector handling |
| **Model Loader** | [Joblib](https://joblib.readthedocs.io/) | Fast serialization and loading of trained models and data |
| **External APIs** | [OMDb API](http://www.omdbapi.com/) & `requests` | Live retrieval of posters, IMDb ratings, and movie details |

---

## 🔍 How It Works

1. **TF-IDF Vectorization:** The textual details (description, genres, cast/crew tags) of each movie are transformed into numerical feature vectors.
2. **K-Nearest Neighbors (KNN):** A similarity space model is constructed. When a user selects a movie, the model calculates the mathematical distance (cosine similarity) to locate the top 5 nearest neighbors in the vector space.
3. **OMDb Enrichment:** The app maps the internal movie IDs to fetch rich metadata from the OMDb API.
4. **Responsive UI Render:** Results are displayed in a clean, multi-column responsive layout complete with interactive visual cards.

---

## 📁 Repository Structure

```text
MACHINE LEARNING/
│
├── Movie_Recommendation.py    # Main Streamlit application entry point
├── requirements.txt           # Python package dependencies
├── movie_data.pkl             # Serialized Pandas DataFrame with movie names and metadata
├── movie_vectors.pkl          # Serialized TF-IDF matrix of movie descriptions
├── movie_model.pkl            # Serialized KNN model for finding similarity
├── clapperboard.png           # Asset: Logo image used in the sidebar header
└── README.md                  # Project documentation (this file)
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed. You will also need an **OMDb API Key** (the project uses a pre-configured key inside the source code, but you can configure your own if needed).

### Installation

1. **Clone or download** this repository.
2. **Install the required packages** using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

### Running the App

Start the Streamlit application by running the following command in your terminal:

```bash
streamlit run Movie_Recommendation.py
```

The app will compile and automatically launch in your default web browser (typically at `http://localhost:8501`).

---

## 📞 Developer Contact

For queries, collaborations, or feedback, feel free to reach out:

- **Email:** [harshrajput74177@gmail.com](mailto:harshrajput74177@gmail.com)
- **Phone:** [+91 74177 09971](tel:7417709971)
