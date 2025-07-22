import streamlit as st
import pandas as pd
import pickle

# Page title
st.title("🎬 Movie Recommender System")

# Load data function
@st.cache_data
def load_data():
    # Load movies directly from movies.pkl
    movies = pd.read_pickle('movies.pkl')
    
    # Load similarity matrix
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    
    return movies, similarity

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    
    return recommended_movies

# Load data
movies, similarity = load_data()

# Movie selection
selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

# Button to get recommendations
if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)
    
    st.write("**Recommended Movies:**")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")
