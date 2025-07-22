import streamlit as st
import pickle
import pandas as pd
import os

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

        recommended_movies = []
        for i in movies_list:
            movie_id = i[0]
            #fetch poster from API
            recommended_movies.append(movies.iloc[i[0]].title)
        return recommended_movies
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return []

# Load data with error handling
@st.cache_data
def load_data():
    try:
        if not os.path.exists('movie_dict.pkl'):
            st.error("movie_dict.pkl not found. Please ensure the file exists in the repository.")
            return None, None
            
        if not os.path.exists('movies.pkl'):
            st.error("movies.pkl not found. Please ensure the file exists in the repository.")
            return None, None
            
        if not os.path.exists('similarity.pkl'):
            st.error("similarity.pkl not found. Please ensure the file exists in the repository.")
            return None, None

        with open('movie_dict.pkl', 'rb') as f:
            movies_dict = pickle.load(f)
        movies = pd.DataFrame(movies_dict)
        
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
            
        return movies, similarity
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Load the data
movies, similarity = load_data()

if movies is None or similarity is None:
    st.stop()

st.title('🎬 Movie Recommender System')

if movies is not None:
    selected_movie_name = st.selectbox(
        'Select a movie you liked:',    
        movies['title'].values)

    if st.button('Show Recommendations'):
        recommendations = recommend(selected_movie_name)
        if recommendations:
            st.subheader('🍿 Movies you might like:')
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        else:
            st.warning("Sorry, couldn't generate recommendations for this movie.")

    # Add app information
    with st.expander("ℹ️ About this app"):
        st.write("This app recommends movies based on similarity to your selected movie.")
        st.write(f"Database contains {len(movies)} movies.")
else:
    st.error("Failed to load movie data. Please check the logs for more details.")
 
