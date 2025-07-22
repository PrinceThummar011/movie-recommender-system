import streamlit as st
import pickle
import pandas as pd
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import re

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

@st.cache_data
def fetch_movie_details(movie_id):
    """Fetch comprehensive movie details from TMDb API"""
    try:
        # Get basic movie details
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits,keywords"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information
            details = {
                'genres': [genre['name'] for genre in data.get('genres', [])],
                'overview': data.get('overview', ''),
                'popularity': data.get('popularity', 0),
                'vote_average': data.get('vote_average', 0),
                'vote_count': data.get('vote_count', 0),
                'runtime': data.get('runtime', 0),
                'release_date': data.get('release_date', ''),
                'keywords': [keyword['name'] for keyword in data.get('keywords', {}).get('keywords', [])],
                'cast': [actor['name'] for actor in data.get('credits', {}).get('cast', [])[:5]],
                'director': next((crew['name'] for crew in data.get('credits', {}).get('crew', []) if crew['job'] == 'Director'), '')
            }
            return details
        return None
    except Exception as e:
        print(f"Error fetching details for movie {movie_id}: {str(e)}")
        return None

def preprocess_text(text):
    """Clean and preprocess text for better similarity calculation"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def create_enhanced_features(movies_df):
    """Create enhanced feature vectors for better recommendations"""
    enhanced_movies = movies_df.copy()
    
    # Clean the tags column
    enhanced_movies['clean_tags'] = enhanced_movies['tegs'].apply(preprocess_text)
    
    # Create TF-IDF vectors with better parameters
    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.8  # Ignore terms that appear in more than 80% of documents
    )
    
    # Fit TF-IDF on the clean tags
    tfidf_matrix = tfidf.fit_transform(enhanced_movies['clean_tags'].fillna(''))
    
    return tfidf_matrix, enhanced_movies

def get_enhanced_recommendations(movie_title, movies_df, num_recommendations=5):
    """Get enhanced movie recommendations using improved similarity calculation"""
    try:
        # Find the movie index
        movie_indices = movies_df[movies_df['title'] == movie_title].index
        if len(movie_indices) == 0:
            return [], []
        
        movie_idx = movie_indices[0]
        
        # Create enhanced feature matrix
        tfidf_matrix, enhanced_movies = create_enhanced_features(movies_df)
        
        # Calculate cosine similarity using linear kernel (more efficient)
        cosine_similarities = linear_kernel(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
        
        # Get movie scores with their indices
        movie_scores = list(enumerate(cosine_similarities))
        
        # Sort movies by similarity (excluding the input movie itself)
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        
        # Get recommended movies and their details
        recommended_movies = []
        recommended_posters = []
        
        for idx, score in movie_scores:
            movie_title = enhanced_movies.iloc[idx]['title']
            movie_id = enhanced_movies.iloc[idx]['movie_id']
            
            recommended_movies.append(movie_title)
            poster_url = fetch_poster(movie_id)
            recommended_posters.append(poster_url)
        
        return recommended_movies, recommended_posters
    
    except Exception as e:
        print(f"Error in enhanced recommendations: {str(e)}")
        return [], []

def get_genre_based_recommendations(movie_title, movies_df, num_recommendations=5):
    """Get recommendations based on genres fetched from TMDb API"""
    try:
        movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
        movie_id = movies_df.iloc[movie_idx]['movie_id']
        
        # Get movie details including genres
        movie_details = fetch_movie_details(movie_id)
        if not movie_details or not movie_details['genres']:
            return get_enhanced_recommendations(movie_title, movies_df, num_recommendations)
        
        # Find movies with similar genres
        recommendations = []
        posters = []
        
        # Get a random sample of movies to check for genre matches
        sample_size = min(500, len(movies_df))  # Check up to 500 movies for efficiency
        sample_indices = np.random.choice(movies_df.index, sample_size, replace=False)
        
        genre_matches = []
        
        for idx in sample_indices:
            if idx == movie_idx:  # Skip the input movie
                continue
                
            candidate_id = movies_df.iloc[idx]['movie_id']
            candidate_details = fetch_movie_details(candidate_id)
            
            if candidate_details and candidate_details['genres']:
                # Calculate genre similarity
                common_genres = set(movie_details['genres']) & set(candidate_details['genres'])
                if common_genres:
                    genre_score = len(common_genres) / len(set(movie_details['genres']) | set(candidate_details['genres']))
                    popularity_score = candidate_details['popularity'] / 1000  # Normalize popularity
                    rating_score = candidate_details['vote_average'] / 10  # Normalize rating
                    
                    # Combined score
                    combined_score = genre_score * 0.5 + popularity_score * 0.3 + rating_score * 0.2
                    
                    genre_matches.append((idx, combined_score, movies_df.iloc[idx]['title']))
        
        # Sort by combined score and get top recommendations
        genre_matches.sort(key=lambda x: x[1], reverse=True)
        
        for idx, score, title in genre_matches[:num_recommendations]:
            movie_id = movies_df.iloc[idx]['movie_id']
            recommendations.append(title)
            poster_url = fetch_poster(movie_id)
            posters.append(poster_url)
        
        # If not enough genre-based matches, fill with content-based recommendations
        if len(recommendations) < num_recommendations:
            content_recs, content_posters = get_enhanced_recommendations(movie_title, movies_df, num_recommendations - len(recommendations))
            recommendations.extend(content_recs)
            posters.extend(content_posters)
        
        return recommendations[:num_recommendations], posters[:num_recommendations]
    
    except Exception as e:
        print(f"Error in genre-based recommendations: {str(e)}")
        return get_enhanced_recommendations(movie_title, movies_df, num_recommendations)

def recommend(movie):
    """Main recommendation function with multiple algorithms"""
    try:
        # Try genre-based recommendations first (higher quality)
        recommendations, posters = get_genre_based_recommendations(movie, movies, 5)
        
        if not recommendations:
            # Fallback to enhanced content-based recommendations
            recommendations, posters = get_enhanced_recommendations(movie, movies, 5)
        
        if not recommendations:
            # Final fallback to original method
            movie_index = movies[movies['title'] == movie].index[0]
            distance = similarity[movie_index]
            movies_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

            recommended_movies = []
            recommended_posters = []
            
            for i in movies_list:
                movie_title = movies.iloc[i[0]].title
                recommended_movies.append(movie_title)
                
                movie_id = movies.iloc[i[0]].movie_id
                poster_url = fetch_poster(movie_id)
                recommended_posters.append(poster_url)
            
            return recommended_movies, recommended_posters
        
        return recommendations, posters
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return [], []

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
        return None, None

# Load the data
movies, similarity = load_data()

if movies is None or similarity is None:
    st.stop()

st.title('🎬 Enhanced Movie Recommender System')

if movies is not None:
    # Add recommendation algorithm selection
    st.sidebar.title("🎯 Recommendation Settings")
    rec_method = st.sidebar.selectbox(
        "Choose recommendation method:",
        ["Smart Hybrid (Recommended)", "Genre-Based", "Content-Based", "Original Method"]
    )
    
    num_recommendations = st.sidebar.slider("Number of recommendations:", 3, 10, 5)
    
    # Movie selection
    selected_movie_name = st.selectbox(
        'Select a movie you liked:',    
        movies['title'].values)
    
    # Show selected movie details
    if selected_movie_name:
        selected_movie_row = movies[movies['title'] == selected_movie_name].iloc[0]
        selected_poster = fetch_poster(selected_movie_row.movie_id)
        movie_details = fetch_movie_details(selected_movie_row.movie_id)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if selected_poster:
                st.image(selected_poster, width=200)
            else:
                st.image("https://via.placeholder.com/300x450/cccccc/969696?text=No+Poster", width=200)
        
        with col2:
            st.markdown(f"### {selected_movie_name}")
            
            if movie_details:
                if movie_details['genres']:
                    st.markdown(f"**Genres:** {', '.join(movie_details['genres'])}")
                if movie_details['vote_average'] > 0:
                    st.markdown(f"**Rating:** ⭐ {movie_details['vote_average']:.1f}/10")
                if movie_details['runtime'] > 0:
                    st.markdown(f"**Runtime:** {movie_details['runtime']} minutes")
                if movie_details['director']:
                    st.markdown(f"**Director:** {movie_details['director']}")
                
                if movie_details['overview']:
                    with st.expander("📖 Plot Overview"):
                        st.write(movie_details['overview'])
            
            st.write("Click the button below to get movie recommendations based on this selection.")

    if st.button('🚀 Get Recommendations'):
        with st.spinner('Finding the best movies for you...'):
            if rec_method == "Genre-Based":
                recommendations, posters = get_genre_based_recommendations(selected_movie_name, movies, num_recommendations)
            elif rec_method == "Content-Based":
                recommendations, posters = get_enhanced_recommendations(selected_movie_name, movies, num_recommendations)
            elif rec_method == "Original Method":
                # Use original similarity matrix
                movie_index = movies[movies['title'] == selected_movie_name].index[0]
                distance = similarity[movie_index]
                movies_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:num_recommendations+1]
                
                recommendations = []
                posters = []
                for i in movies_list:
                    movie_title = movies.iloc[i[0]].title
                    recommendations.append(movie_title)
                    movie_id = movies.iloc[i[0]].movie_id
                    poster_url = fetch_poster(movie_id)
                    posters.append(poster_url)
            else:  # Smart Hybrid
                recommendations, posters = recommend(selected_movie_name)
                
        if recommendations:
            st.subheader('🍿 Movies you might like:')
            
            # Create responsive columns
            cols_per_row = min(5, len(recommendations))
            cols = st.columns(cols_per_row)
            
            for i, (movie, poster_url) in enumerate(zip(recommendations, posters)):
                col_idx = i % cols_per_row
                
                with cols[col_idx]:
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        st.image("https://via.placeholder.com/300x450/cccccc/969696?text=No+Poster", width=150)
                    
                    st.markdown(f"**{movie}**")
                    
                    # Get additional details for recommended movies
                    rec_movie_row = movies[movies['title'] == movie]
                    if not rec_movie_row.empty:
                        rec_movie_id = rec_movie_row.iloc[0].movie_id
                        rec_details = fetch_movie_details(rec_movie_id)
                        
                        if rec_details and rec_details['vote_average'] > 0:
                            st.markdown(f"⭐ {rec_details['vote_average']:.1f}")
                        
                        if rec_details and rec_details['genres']:
                            # Show first 2 genres
                            genres_display = ', '.join(rec_details['genres'][:2])
                            st.markdown(f"*{genres_display}*")
        else:
            st.warning("Sorry, couldn't generate recommendations for this movie. Try selecting a different movie or recommendation method.")

    # Add app information
    with st.expander("ℹ️ About this Enhanced Recommender"):
        st.write("This improved movie recommender uses multiple algorithms:")
        st.write("• **Smart Hybrid**: Combines genre-based and content-based recommendations")
        st.write("• **Genre-Based**: Uses TMDb API to find movies with similar genres, popularity, and ratings")
        st.write("• **Content-Based**: Enhanced TF-IDF analysis of movie descriptions and metadata")
        st.write("• **Original Method**: Simple cosine similarity on basic features")
        st.write(f"Database contains {len(movies)} movies.")
        st.write("Movie details and posters are fetched from The Movie Database (TMDb).")
else:
    st.error("Failed to load movie data. Please check the logs for more details.")
 
