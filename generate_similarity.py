import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

print("Loading existing movie data...")
# Load the existing movies data
with open('movies.pkl', 'rb') as f:
    movies = pickle.load(f)

print(f"Loaded {len(movies)} movies")

# Check if we have the required columns
required_columns = ['title', 'overview', 'genres', 'keywords', 'cast', 'crew']
missing_columns = [col for col in required_columns if col not in movies.columns]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
    print("Creating a simple similarity matrix based on available data...")
    
    # Create a simple similarity matrix based on title and overview if available
    if 'overview' in movies.columns:
        # Fill NaN values
        movies['overview'] = movies['overview'].fillna('')
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies['overview']).toarray()
    else:
        # If no overview, create identity matrix as fallback
        n_movies = len(movies)
        vectors = np.eye(n_movies)
    
    similarity = cosine_similarity(vectors)
    
else:
    print("All required columns found. Creating full similarity matrix...")
    
    # Process the data as in the notebook
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Clean the data
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    
    # Create tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())
    
    # Create similarity matrix
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

print(f"Similarity matrix shape: {similarity.shape}")

# Save the similarity matrix
print("Saving similarity matrix...")
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print("similarity.pkl created successfully!")
print(f"File size: {len(pickle.dumps(similarity)) / 1024 / 1024:.2f} MB")
