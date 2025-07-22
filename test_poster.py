import requests

# Test the API
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def test_poster_fetch(movie_id):
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Movie: {data.get('title', 'Unknown')}")
            poster_path = data.get('poster_path')
            if poster_path:
                poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
                print(f"Poster URL: {poster_url}")
                return poster_url
            else:
                print("No poster available")
        else:
            print(f"API Error: {response.status_code}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test with Avatar (movie_id from your dataset)
print("Testing poster fetch for Avatar (ID: 19995)")
test_poster_fetch(19995)
