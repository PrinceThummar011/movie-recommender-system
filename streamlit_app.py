import streamlit as st
import pandas as pd
import pickle
import requests
import os

# Configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load movie data and similarity matrix"""
    try:
        # Load movies data
        if os.path.exists('movies.pkl'):
            movies = pd.read_pickle('movies.pkl')
        elif os.path.exists('movie_dict.pkl'):
            with open('movie_dict.pkl', 'rb') as f:
                movies_dict = pickle.load(f)
            movies = pd.DataFrame(movies_dict)
        else:
            st.error("❌ No movie data files found!")
            return None, None
        
        # Load similarity matrix
        if os.path.exists('similarity_compressed.pkl.gz'):
            import gzip
            with gzip.open('similarity_compressed.pkl.gz', 'rb') as f:
                similarity = pickle.load(f)
        elif os.path.exists('similarity.pkl'):
            with open('similarity.pkl', 'rb') as f:
                similarity = pickle.load(f)
        else:
            st.error("❌ No similarity matrix found!")
            return None, None
            
        return movies, similarity
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

def recommend(movie, movies, similarity):
    """Get movie recommendations"""
    try:
        # Find movie index
        movie_index = movies[movies['title'] == movie].index
        
        if len(movie_index) == 0:
            return []
        
        movie_index = movie_index[0]
        
        # Get similarity scores
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
        
        # Get top 5 recommendations (excluding the input movie)
        recommended_movies = []
        for i, score in movies_list[1:6]:
            recommended_movies.append(movies.iloc[i]['title'])
            
        return recommended_movies
        
    except Exception as e:
        st.error(f"❌ Error in recommendation: {str(e)}")
        return []

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🎬 Movie Recommender System")
    st.markdown("**Discover movies you'll love based on your preferences!**")
    
    # Load data
    with st.spinner("Loading movie data..."):
        movies, similarity = load_data()
    
    if movies is None or similarity is None:
        st.stop()
    
    st.success(f"✅ Loaded {len(movies)} movies successfully!")
    
    # Sidebar
    st.sidebar.header("🎯 Movie Recommendation")
    st.sidebar.markdown("Select a movie you like and get personalized recommendations!")
    
    # Movie selection
    selected_movie = st.selectbox(
        "🎬 Choose a movie you enjoyed:",
        options=sorted(movies['title'].unique()),
        index=0
    )
    
    # Number of recommendations
    num_recommendations = st.slider(
        "📊 Number of recommendations:",
        min_value=3,
        max_value=10,
        value=5
    )
    
    # Get recommendations button
    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
            recommendations = recommend(selected_movie, movies, similarity)
            
            if recommendations:
                st.success(f"🎯 Here are {len(recommendations)} movies you might enjoy:")
                
                # Display recommendations
                cols = st.columns(min(len(recommendations), 3))
                
                for i, movie in enumerate(recommendations[:num_recommendations]):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="
                            padding: 1rem;
                            border-radius: 10px;
                            border: 1px solid #e0e0e0;
                            margin: 0.5rem 0;
                            background: linear-gradient(45deg, #f0f2f6, #ffffff);
                        ">
                            <h4 style="margin: 0; color: #1f4e79;">
                                {i+1}. {movie}
                            </h4>
                            <p style="margin: 0.5rem 0 0 0; color: #666;">
                                🎬 Recommended for you
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            else:
                st.error("❌ No recommendations found. Try selecting a different movie.")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📊 About This App")
    st.info("""
    This Movie Recommender System uses **Content-Based Filtering** and **Cosine Similarity** 
    to suggest movies similar to your preferences. The recommendations are based on:
    - Movie genres
    - Plot overview
    - Cast and crew
    - Keywords and themes
    """)
    
    # Statistics
    if movies is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎬 Total Movies", f"{len(movies):,}")
        
        with col2:
            if 'genres' in movies.columns:
                unique_genres = len(set([genre for genres in movies['genres'].str.split('|').fillna([]) for genre in genres]))
                st.metric("🎭 Genres", unique_genres)
            else:
                st.metric("🎭 Features", f"{similarity.shape[1]:,}")
        
        with col3:
            st.metric("🤖 Algorithm", "Cosine Similarity")

if __name__ == "__main__":
    main()
