import streamlit as st
import pickle
import pandas as pd
import os
import requests

# TMDb API configuration
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # Free API key (you can get your own)
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def fetch_poster(movie_id):
    """Fetch movie poster from TMDb API"""
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        return None
    except Exception as e:
        print(f"Error fetching poster for movie {movie_id}: {str(e)}")
        return None

<<<<<<< HEAD
def recommend(movie, movies, similarity):
    try:
        # DEBUG: Show input
        st.write(f"🔍 Input movie: '{movie}'")
        st.write(f"📊 Movies dataframe shape: {movies.shape}")
        st.write(f"📊 Similarity matrix shape: {similarity.shape}")
        
        # Find movie index - CRITICAL FIX
        movie_index = movies[movies['title'] == movie].index
        
        if len(movie_index) == 0:
            st.error(f"Movie '{movie}' not found in database!")
            return [], []
        
        movie_index = movie_index[0]
        st.write(f"✅ Found movie at index: {movie_index}")
        
        # Get similarity distances for THIS specific movie
        distances = similarity[movie_index]  # This should be different for each movie!
        
        # DEBUG: Show similarity scores
        st.write(f"🔍 Similarity scores sample: {distances[:5]}")
        st.write(f"🔍 Max similarity: {distances.max():.3f}")
        st.write(f"🔍 Min similarity: {distances.min():.3f}")
        
        # Sort movies by similarity
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
        
        # DEBUG: Show top similar movies with scores
        st.write("🔍 Top 6 similar movies:")
        for i, (idx, score) in enumerate(movies_list[:6]):
            debug_title = movies.iloc[idx]['title']
            st.write(f"   {i+1}. {debug_title} (Score: {score:.3f}, Index: {idx})")
        
        recommended_movies = []
        recommended_posters = []
        
        # Get top 5 recommendations (excluding the input movie)
        for i, score in movies_list[1:6]:  # Skip first (input movie)
            recommended_movies.append(movies.iloc[i]['title'])
            
            # Try to fetch poster (or use placeholder)
            try:
                movie_id = movies.iloc[i]['movie_id']
                poster_url = fetch_poster(movie_id)
                recommended_posters.append(poster_url)
            except:
                recommended_posters.append(None)
            
        return recommended_movies, recommended_posters
        
    except Exception as e:
        st.error(f"❌ Error in recommendation: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
=======
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

        recommended_movies = []
        recommended_posters = []
        
        for i in movies_list:
            # Get movie title
            movie_title = movies.iloc[i[0]].title
            recommended_movies.append(movie_title)
            
            # Get movie poster
            movie_id = movies.iloc[i[0]].movie_id
            poster_url = fetch_poster(movie_id)
            recommended_posters.append(poster_url)
            
        return recommended_movies, recommended_posters
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
>>>>>>> 1550e7c27efa881349f10798c642593c633c1634
        return [], []

# Load data with error handling
@st.cache_data
def load_data():
    try:
<<<<<<< HEAD
        # DEBUG: Add these print statements
        st.write("🔍 Loading data files...")
        
        # Check if using correct files
        if os.path.exists('movie_dict.pkl'):
            with open('movie_dict.pkl', 'rb') as f:
                movies_dict = pickle.load(f)
            movies = pd.DataFrame(movies_dict)
            st.write(f"✅ Loaded {len(movies)} movies from movie_dict.pkl")
        else:
            # Try loading movies.pkl directly
            movies = pd.read_pickle('movies.pkl')
            st.write(f"✅ Loaded {len(movies)} movies from movies.pkl")
        
        # Check similarity matrix
        if os.path.exists('similarity_compressed.pkl.gz'):
            import gzip
            with gzip.open('similarity_compressed.pkl.gz', 'rb') as f:
                similarity = pickle.load(f)
            st.write("✅ Loaded compressed similarity matrix")
        else:
            with open('similarity.pkl', 'rb') as f:
                similarity = pickle.load(f)
            st.write(f"✅ Loaded similarity matrix: {similarity.shape}")
        
        # DEBUG: Show data sample
        st.write("📊 Sample movies:", movies['title'].head().tolist())
        
        return movies, similarity
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
=======
        if not os.path.exists('movie_dict.pkl'):
            st.error("movie_dict.pkl not found. Please ensure the file exists in the repository.")
            return None, None
            
        if not os.path.exists('movies.pkl'):
            st.error("movies.pkl not found. Please ensure the file exists in the repository.")
            return None, None
            
        # Try loading compressed similarity first, then fallback to regular
        similarity = None
        if os.path.exists('similarity_compressed.pkl.gz'):
            import gzip
            from scipy import sparse
            with gzip.open('similarity_compressed.pkl.gz', 'rb') as f:
                similarity_sparse = pickle.load(f)
                similarity = similarity_sparse.toarray()
        elif os.path.exists('similarity.pkl'):
            with open('similarity.pkl', 'rb') as f:
                similarity = pickle.load(f)
        else:
            st.error("No similarity file found. Please ensure similarity.pkl or similarity_compressed.pkl.gz exists.")
            return None, None

        with open('movie_dict.pkl', 'rb') as f:
            movies_dict = pickle.load(f)
        movies = pd.DataFrame(movies_dict)
            
        return movies, similarity
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
>>>>>>> 1550e7c27efa881349f10798c642593c633c1634
        return None, None

# Load the data
movies, similarity = load_data()

if movies is None or similarity is None:
    st.stop()

st.title('🎬 Movie Recommender System')

if movies is not None:
    # Movie selection
    selected_movie_name = st.selectbox(
        'Select a movie you liked:',    
        movies['title'].values)
    
    # Show selected movie poster
    if selected_movie_name:
        selected_movie_row = movies[movies['title'] == selected_movie_name].iloc[0]
        selected_poster = fetch_poster(selected_movie_row.movie_id)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if selected_poster:
                st.image(selected_poster, width=200)
            else:
                st.image("https://via.placeholder.com/300x450/cccccc/969696?text=No+Poster", width=200)
        
        with col2:
            st.markdown(f"### {selected_movie_name}")
            st.write("Click the button below to get movie recommendations based on this selection.")

    if st.button('Show Recommendations'):
        with st.spinner('Finding similar movies...'):
<<<<<<< HEAD
            recommendations, posters = recommend(selected_movie_name, movies, similarity)
=======
            recommendations, posters = recommend(selected_movie_name)
>>>>>>> 1550e7c27efa881349f10798c642593c633c1634
            
        if recommendations:
            st.subheader('🍿 Movies you might like:')
            
            # Create columns for movie display
            cols = st.columns(5)
            
            for i, (movie, poster_url) in enumerate(zip(recommendations, posters)):
                with cols[i]:
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        st.image("https://via.placeholder.com/300x450/cccccc/969696?text=No+Poster", width=150)
                    
                    st.markdown(f"**{movie}**")
        else:
            st.warning("Sorry, couldn't generate recommendations for this movie.")

    # Add app information
    with st.expander("ℹ️ About this app"):
        st.write("This app recommends movies based on similarity to your selected movie.")
        st.write(f"Database contains {len(movies)} movies.")
        st.write("Posters are fetched from The Movie Database (TMDb).")
else:
    st.error("Failed to load movie data. Please check the logs for more details.")
<<<<<<< HEAD

# Add this debug section BEFORE the main app
if st.checkbox("🔍 Debug Mode"):
    st.header("🔍 Debug Information")
    
    movies, similarity = load_data()
    
    if movies is not None:
        st.write("**Data loaded successfully!**")
        
        # Test with different movies
        test_movies = ['Avatar', 'Titanic', 'The Dark Knight', 'Inception']
        
        for test_movie in test_movies:
            if test_movie in movies['title'].values:
                st.write(f"\n**Testing: {test_movie}**")
                
                # Find index
                movie_idx = movies[movies['title'] == test_movie].index[0]
                st.write(f"Index: {movie_idx}")
                
                # Show similarity scores for this movie
                sim_scores = similarity[movie_idx]
                top_indices = sim_scores.argsort()[::-1][:6]
                
                st.write("Top similar movies:")
                for i, idx in enumerate(top_indices):
                    title = movies.iloc[idx]['title']
                    score = sim_scores[idx]
                    st.write(f"{i+1}. {title} ({score:.3f})")
                
                st.write("---")

# Add this button to clear cache
if st.button("🔄 Clear Cache & Reload"):
    st.cache_data.clear()
    st.experimental_rerun()

=======
 
>>>>>>> 1550e7c27efa881349f10798c642593c633c1634
