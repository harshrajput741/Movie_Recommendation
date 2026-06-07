# ==========================================
# MOVIE RECOMMENDATION SYSTEM - MAIN APP FILE
# ==========================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Movie AI Recommender", page_icon="🎬", layout="wide")

# 1a. OMDB API CALL CACHING
@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    try:
        url = f'http://www.omdbapi.com/?i={movie_id}&apikey=7918b46d'
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}

# 1b. GLOBAL PREMIUM DARK THEME CSS INJECTION
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* Global resets and font family overrides */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0b0914 !important;
    background-image: radial-gradient(circle at 10% 20%, rgba(138, 35, 135, 0.08) 0%, transparent 40%),
                      radial-gradient(circle at 90% 80%, rgba(242, 113, 33, 0.06) 0%, transparent 40%) !important;
    color: #f3f4f6 !important;
}
/* Custom premium scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0b0914;
}
::-webkit-scrollbar-thumb {
    background: rgba(233, 64, 87, 0.3);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(233, 64, 87, 0.5);
}
/* Page container spacing controls */
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}
/* Custom styled Streamlit selectbox component */
div[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}
div[data-baseweb="select"]:hover {
    border-color: rgba(233, 64, 87, 0.4) !important;
    box-shadow: 0 0 12px rgba(233, 64, 87, 0.15) !important;
}
div[data-baseweb="select"] * {
    color: white !important;
    background-color: transparent !important;
}
ul[role="listbox"] {
    background-color: #121020 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
}
ul[role="listbox"] li {
    color: #e5e7eb !important;
    transition: background-color 0.2s;
}
ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
    background-color: rgba(233, 64, 87, 0.15) !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# 2. LOAD TRAINED MODELS & DATASETS
X = joblib.load('movie_vectors.pkl')      # TF-IDF vector matrix containing movie descriptions
df = joblib.load('movie_data.pkl')        # Pandas DataFrame with movie titles, IDs, and details
model = joblib.load('movie_model.pkl')    # K-Nearest Neighbors similarity model

# 3. SIDEBAR NAVIGATION & INFO PANEL
# Defines the sidebar layouts, custom styles, project metadata, and contact info.
with st.sidebar:
    # Sidebar Header branding (Logo and Title)
    col1, col2 = st.columns([1.2, 3])
    with col1:
        st.image("clapperboard.png", width=48)
    with col2:
        st.markdown("""
        <div style="
            height:55px;
            display:flex;
            align-items:center;
            color:#E94057;
            font-size:30px;
            font-weight:800;
            letter-spacing: -0.5px;
        ">
            Movie AI
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Custom CSS Styling for Sidebar Cards to create a glassmorphism look
    st.markdown("""
    <style>
    .sidebar-section {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    .sidebar-section:hover {
        border-color: rgba(233, 64, 87, 0.4);
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(138, 35, 135, 0.15);
    }
    .sidebar-header {
        font-size: 15px;
        font-weight: 700;
        color: #E94057;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: 0.5px;
    }
    .sidebar-content {
        color: #9ca3af;
        font-size: 13px;
        line-height: 1.5;
    }
    .sidebar-content ul {
        margin: 0;
        padding-left: 20px;
    }
    .contact-item {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 4px 8px;
        margin-bottom: 8px;
        color: #9ca3af;
        font-size: 13px;
    }
    .contact-item a {
        color: #E94057;
        text-decoration: none;
        transition: color 0.2s;
        font-weight: 600;
        word-break: break-all;
    }
    .contact-item a:hover {
        color: #F27121;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Info card describing the project goal
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-header">📊 Project Info</div>
        <div class="sidebar-content">
            Objective of this project is to recommend similar movies using a Content-Based Filtering approach with vector embeddings.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech Stack & Libraries list
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-header">🛠 Tech Stack & ML Libs</div>
        <div class="sidebar-content">
            <ul style="margin: 0; padding-left: 18px;">
                <li><b>Streamlit</b> (UI Dashboard)</li>
                <li><b>Scikit-Learn (Sklearn)</b>:
                    <ul style="margin: 4px 0; padding-left: 15px; list-style-type: circle;">
                        <li><i>TfidfVectorizer</i> (NLP/Feature extraction)</li>
                        <li><i>NearestNeighbors</i> (KNN Similarity model)</li>
                    </ul>
                </li>
                <li><b>Pandas</b> (Dataset manipulation)</li>
                <li><b>NumPy</b> (Vector mathematics)</li>
                <li><b>Joblib</b> (Model serialization/loading)</li>
                <li><b>Requests & OMDb API</b> (Poster retrieval)</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Developer Contact card with telephone and mailto links
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-header">📞 Contact Info</div>
        <div class="sidebar-content">
            <div class="contact-item">
                📱 <b>Phone:</b> <a href="tel:7417709971">7417709971</a>
            </div>
            <div class="contact-item">
                ✉️ <b>Email:</b> <a href="mailto:harshrajput74177@gmail.com">harshrajput74177@gmail.com</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 4. HEADER COMPONENT STYLE & HTML
st.markdown("""
<style>
.modern-header {
    background: linear-gradient(
        135deg,
        rgba(138, 35, 135, 0.1),
        rgba(233, 64, 87, 0.08),
        rgba(242, 113, 33, 0.05)
    );
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 25px 20px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}
.modern-header h1 {
    font-size: 44px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #8A2387, #E94057, #F27121);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.8px;
}
.modern-header p {
    color: #9ca3af;
    font-size: 16px;
    margin-top: 8px;
    margin-bottom: 0;
    font-weight: 400;
}
</style>
<div class="modern-header">
    <h1>Discover Movies You'll Love 🎬</h1>
    <p>AI-powered recommendation engine based on description similarity</p>
</div>
""", unsafe_allow_html=True)
st.write('\n')

# 5. DROPDOWN SUBHEADER STYLE & HTML
st.markdown("""
    <style>
    .Single-header {
        background: linear-gradient(
            145deg,
            rgba(138, 35, 135, 0.08),
            rgba(233, 64, 87, 0.05)
        );
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 12px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    .Single-header h1 {
        font-size: 20px;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #8A2387, #E94057);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    </style>
    <div class="Single-header">
        <h1>Select a Movie to get Recommendations</h1>
    </div>
    """, unsafe_allow_html=True)
st.write('\n')

# 6. MOVIE RECOMMENDATION CARD SYSTEM STYLE & HTML
st.markdown("""
<style>
.movie-card {
    position: relative;
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    border-radius: 16px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
}
.movie-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: rgba(233, 64, 87, 0.5);
    background: rgba(255, 255, 255, 0.05);
    box-shadow: 0 15px 30px rgba(138, 35, 135, 0.20), 
                0 0 15px rgba(233, 64, 87, 0.15);
}
.movie-poster {
    width: 100%;
    height: 250px;
    object-fit: cover;
    border-radius: 12px;
    transition: all 0.4s ease;
}
.movie-card:hover .movie-poster {
    filter: brightness(1.1);
    transform: scale(1.01);
}
.movie-title {
    color: #ffffff;
    font-weight: 700;
    font-size: 14px;
    margin-top: 10px;
    text-align: center;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    height: 38px;
    line-height: 1.35;
}
.movie-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
    font-size: 11px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    padding-top: 8px;
}
.movie-rating {
    color: #FFD700;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 3px;
}
.movie-year {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #9ca3af;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
}
.recommend-header {
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg, #8A2387, #E94057, #F27121);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 25px 0 15px 0;
    letter-spacing: -0.5px;
}
.movie-card-link {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
}
.movie-card-link:hover, .movie-card-link:visited, .movie-card-link:active {
    text-decoration: none !important;
    color: inherit !important;
}
</style>
""", unsafe_allow_html=True)

# 7. MAIN LOGIC - USER INTERACTION & RECOMMENDATIONS
# Displays movie selection dropdown. When selected, runs model to search and query OMDB API for 5 suggestions.
mvname = st.selectbox('Select a movie',['Choose a Movie'] + list(df['name']))

if mvname != 'Choose a Movie':
    # Locates the index of the selected movie in dataframe
    movie = df[df['name'] == mvname]
    if not movie.empty:
        # --- A. SELECTED MOVIE DETAILS CARD ---
        selected_id = movie['movie_id'].values[0]
        selected_desc = movie['content'].values[0]
        # Clean description text by splitting out bracketed lists (cast, crew, genre metadata tags)
        if " [" in selected_desc:
            selected_desc = selected_desc.split(" [")[0].strip()
        
        # Fetch details from cached OMDb function
        selected_details = fetch_movie_details(selected_id)
        selected_poster = selected_details.get('Poster')
        selected_rating = selected_details.get('imdbRating', 'N/A')
        selected_year = selected_details.get('Year', 'N/A')
        selected_genre = selected_details.get('Genre', 'N/A')
        selected_director = selected_details.get('Director', 'N/A')
        selected_runtime = selected_details.get('Runtime', 'N/A')
        
        if not selected_poster or selected_poster == "N/A":
            selected_poster = "https://via.placeholder.com/300x450?text=No+Poster"
            
        # Display selected movie details card
        st.markdown(f"""
        <style>
        .selected-movie-container {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 24px;
            margin-top: 10px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            display: flex;
            gap: 24px;
            align-items: flex-start;
        }}
        .selected-poster-container {{
            flex-shrink: 0;
            width: 180px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .selected-poster {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .selected-info {{
            flex-grow: 1;
        }}
        .selected-title {{
            font-size: 28px;
            font-weight: 800;
            margin: 0 0 10px 0;
            color: #ffffff;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #ffffff, #e5e7eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .selected-badges {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-rating {{
            background: rgba(255, 215, 0, 0.12);
            color: #FFD700;
            border: 1px solid rgba(255, 215, 0, 0.25);
        }}
        .badge-year {{
            background: rgba(255, 255, 255, 0.06);
            color: #d1d5db;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .badge-runtime {{
            background: rgba(233, 64, 87, 0.08);
            color: #E94057;
            border: 1px solid rgba(233, 64, 87, 0.15);
        }}
        .badge-genre {{
            background: rgba(138, 35, 135, 0.12);
            color: #c084fc;
            border: 1px solid rgba(138, 35, 135, 0.2);
        }}
        .selected-section-title {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #E94057;
            margin-bottom: 6px;
        }}
        .selected-overview {{
            color: #d1d5db;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 16px;
        }}
        .selected-director {{
            color: #9ca3af;
            font-size: 13px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            padding-top: 12px;
        }}
        .selected-director b {{
            color: #ffffff;
        }}
        @media (max-width: 768px) {{
            .selected-movie-container {{
                flex-direction: column;
                align-items: center;
                text-align: center;
            }}
            .selected-badges {{
                justify-content: center;
            }}
            .selected-poster-container {{
                width: 150px;
            }}
        }}
        </style>
        
        <div class="selected-movie-container">
            <div class="selected-poster-container">
                <a href="https://www.imdb.com/title/{selected_id}/" target="_blank" title="View on IMDb">
                    <img class="selected-poster" src="{selected_poster}" alt="{mvname}">
                </a>
            </div>
            <div class="selected-info">
                <h2 class="selected-title">{mvname}</h2>
                <div class="selected-badges">
                    <span class="badge badge-rating">⭐ {selected_rating}</span>
                    <span class="badge badge-year">{selected_year}</span>
                    <span class="badge badge-runtime">⏱️ {selected_runtime}</span>
                    <span class="badge badge-genre">🎬 {selected_genre}</span>
                </div>
                <div class="selected-section-title">Overview</div>
                <p class="selected-overview">{selected_desc}</p>
                <div class="selected-director">Directed by: <b>{selected_director}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Displays the subheader for recommended section
        st.markdown(
            '<div class="recommend-header">🎯 Recommended For You</div>',
            unsafe_allow_html=True)

        index = movie.index[0]
        vector = X[index] # Gets the TF-IDF representation of the movie content
        
        # Uses KNN model to find 6 nearest movie vectors (1st is the selected movie itself, 2nd-6th are recommendations)
        distances, indexes = model.kneighbors(vector, n_neighbors=6)
        
        # Creates 5 side-by-side columns to render the recommended movies
        cols = st.columns(5)
        
        # Iterates through the top 5 closest neighbors (excluding the selected movie at indexes[0][0])
        for idx, i in enumerate(indexes[0][1:6]):
            movie_name = df.loc[i, 'name']
            mid = df.loc[i, 'movie_id']
            
            # Fetch details from cached OMDb function
            details = fetch_movie_details(mid)
            poster = details.get('Poster')
            rating = details.get('imdbRating', 'N/A')
            year = details.get('Year', 'N/A')
            
            # If poster URL is empty or OMDb doesn't have it, load a placeholder
            if not poster or poster == "N/A":
                poster = "https://via.placeholder.com/300x450?text=No+Poster"
            
            # Renders each recommended movie card inside its respective column
            with cols[idx]: 
                st.markdown(
                    f"""
                    <a href="https://www.imdb.com/title/{mid}/" target="_blank" class="movie-card-link" title="Click to view details on IMDb">
                        <div class="movie-card">
                            <img class="movie-poster" src="{poster}" alt="{movie_name}">
                            <div class="movie-title">{movie_name}</div>
                            <div class="movie-meta">
                                <span class="movie-rating">⭐ {rating}</span>
                                <span class="movie-year">{year}</span>
                            </div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )