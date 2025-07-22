import streamlit as st
import pandas as pd
import pickle
import os
import gzip

# Page configuration
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
    """Get movie recommendations using the exact same logic as your notebook"""
    try:
        # Find movie index - exactly like your notebook
        movie_index = movies[movies['title'] == movie].index[0]
        
        # Get similarity distances - exactly like your notebook
        distance = similarity[movie_index]
        
        # Sort and get top 5 - exactly like your notebook
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        
        # Return movie titles - exactly like your notebook
        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(movies.iloc[i[0]].title)
            
        return recommended_movies
        
    except Exception as e:
        st.error(f"❌ Error in recommendation: {str(e)}")
        return []

def main():
    """Main application"""
    
    # Title
    st.title("🎬 Movie Recommender System")
    st.markdown("**Get movie recommendations from our database of 4,803 movies!**")
    
    # Load data
    movies, similarity = load_data()
    
    if movies is None or similarity is None:
        st.stop()
    
    # Show dataset info
    st.success(f"✅ Loaded {len(movies)} movies from your trained model")
    
    # Show some sample movies from YOUR dataset
    with st.expander("📋 Sample movies in our database"):
        sample_movies = movies['title'].head(20).tolist()
        st.write(", ".join(sample_movies))
    
    # Movie selection - ONLY from your actual dataset
    selected_movie = st.selectbox(
        "🎬 Choose a movie from our database:",
        options=sorted(movies['title'].tolist()),  # Only YOUR movies
        help=f"Select from {len(movies)} movies in the database"
    )
    
    # Show selected movie info
    if selected_movie:
        movie_info = movies[movies['title'] == selected_movie].iloc[0]
        st.info(f"**Selected:** {selected_movie} (Movie ID: {movie_info['movie_id']})")
    
    # Get recommendations button
    if st.button("🚀 Show Recommendations", type="primary"):
        with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
            recommendations = recommend(selected_movie, movies, similarity)
            
            if recommendations:
                st.subheader("🎯 Movies you might like:")
                
                # Display recommendations in a nice format
                for i, movie in enumerate(recommendations, 1):
                    # Get movie info from YOUR dataset
                    movie_info = movies[movies['title'] == movie].iloc[0]
                    
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            st.markdown(f"### {i}")
                        
                        with col2:
                            st.markdown(f"**{movie}**")
                            st.caption(f"Movie ID: {movie_info['movie_id']}")
                        
                        st.divider()
                        
            else:
                st.error("❌ No recommendations found!")
    
    # Dataset statistics
    st.markdown("---")
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎬 Total Movies", f"{len(movies):,}")
    
    with col2:
        st.metric("🔢 Movie IDs Range", f"{movies['movie_id'].min()} - {movies['movie_id'].max()}")
    
    with col3:
        st.metric("🤖 Algorithm", "Cosine Similarity")
    
    # Show actual movie titles from your dataset
    if st.checkbox("🔍 Browse all movies in database"):
        st.subheader("All Movies in Your Dataset:")
        
        # Search functionality
        search_term = st.text_input("Search movies:", placeholder="Type to search...")
        
        if search_term:
            filtered_movies = movies[movies['title'].str.contains(search_term, case=False, na=False)]
            st.write(f"Found {len(filtered_movies)} movies matching '{search_term}':")
            for movie in filtered_movies['title'].tolist():
                st.write(f"• {movie}")
        else:
            # Show all movies in pages
            movies_per_page = 50
            total_pages = len(movies) // movies_per_page + 1
            
            page = st.selectbox("Select page:", range(1, total_pages + 1))
            
            start_idx = (page - 1) * movies_per_page
            end_idx = start_idx + movies_per_page
            
            page_movies = movies['title'].iloc[start_idx:end_idx].tolist()
            
            st.write(f"**Page {page} of {total_pages}** (Movies {start_idx + 1} - {min(end_idx, len(movies))}):")
            
            for movie in page_movies:
                st.write(f"• {movie}")

if __name__ == "__main__":
    main()
