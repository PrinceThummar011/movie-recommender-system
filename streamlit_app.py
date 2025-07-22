import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommender System")

@st.cache_data
def load_data():
    try:
        # Load exactly as your notebook
        with open('movie_dict.pkl', 'rb') as f:
            movie_dict = pickle.load(f)
        movies = pd.DataFrame(movie_dict)
        
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        
        return movies, similarity
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def recommend(movie):
    """Exact same function as your notebook"""
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommendations = []
        for i in movies_list:
            recommendations.append(new_df.iloc[i[0]].title)
        
        return recommendations
    except:
        return []

# Load data
new_df, similarity = load_data()

if new_df is not None:
    # Movie selection
    selected_movie = st.selectbox("Select a movie:", new_df['title'].values)
    
    if st.button("Show Recommendations"):
        recommendations = recommend(selected_movie)
        
        if recommendations:
            st.write("**Recommendations:**")
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        else:
            st.error("No recommendations found!")
