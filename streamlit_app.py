import streamlit as st
import pandas as pd
import pickle
import requests

st.title("🎬 Movie Recommender")

# TMDb API configuration
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # Free API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def load_data():
    movies = pd.read_pickle('movies.pkl')
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    return movies, similarity

def fetch_poster(movie_id):
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        poster_path = data.get('poster_path')
        if poster_path:
            return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        return None
    except:
        return None

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    
    for i in movies_list:
        movie_title = movies.iloc[i[0]].title
        movie_id = movies.iloc[i[0]].movie_id
        
        recommended_movies.append(movie_title)
        poster = fetch_poster(movie_id)
        recommended_posters.append(poster)
    
    return recommended_movies, recommended_posters

# Load data
movies, similarity = load_data()

selected_movie = st.selectbox("Select a movie:", movies['title'].values)

if st.button("Recommend"):
    movies_list, posters_list = recommend(selected_movie)
    
    # Display in columns
    cols = st.columns(5)
    
    for i in range(len(movies_list)):
        with cols[i]:
            st.subheader(f"{i+1}")
            
            # Show poster if available
            if posters_list[i]:
                st.image(posters_list[i], width=150)
            else:
                st.write("🎬 No Poster")
            
            # Show movie name
            st.write(f"**{movies_list[i]}**")