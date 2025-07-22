# 🎬 Movie Recommender System

This is an intelligent movie recommendation system built using Python and machine learning. It recommends movies based on content similarity using advanced filtering techniques.

## 🌟 Live Demo
**🚀 Try the app live:** [https://movie-recommender-system-by-prince.streamlit.app/](https://movie-recommender-system-by-prince.streamlit.app/)

## 📌 Features

- Smart movie recommendations based on content similarity
- Interactive web interface with movie posters
- Multiple recommendation algorithms (Content-based, Genre-based, Hybrid)
- Movie details including ratings, genres, and overview
- Responsive design with beautiful UI
- Real-time poster fetching from TMDb API

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Web app framework
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **TMDb API** - Movie data and posters
- **Pickle** - Data serialization
- **Requests** - API calls

## 📁 Project Structure

```
movie-recommender-system/
├── streamlit_app.py           # Main Streamlit web application
├── movie_dict.pkl            # Preprocessed movie metadata
├── movies.pkl               # Movie dataset
├── similarity_compressed.pkl.gz  # Compressed similarity matrix
├── movie_recommender.ipynb   # Jupyter notebook for development
├── requirements.txt         # Python dependencies
├── dataset/                # Dataset information
└── README.md               # Project documentation
```

## 🚀 How to Run the Project

### Option 1: Try the Live App (Recommended)
🌐 **Visit the live app:** [https://movie-recommender-system-by-prince.streamlit.app/](https://movie-recommender-system-by-prince.streamlit.app/)

No installation required! Just click the link and start getting movie recommendations instantly.

### Option 2: Run Locally

#### 1. Clone the repository
```bash
git clone https://github.com/PrinceThummar011/movie-recommender-system.git
cd movie-recommender-system
```

#### 2. Install required libraries
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas scikit-learn numpy scipy requests
```

#### 3. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

#### 4. Use the app
- Select a movie from the dropdown
- Click on "Show Recommendations"  
- You'll see 5 similar movies with posters and details

## 📊 Dataset
This project uses the TMDb (The Movie Database) dataset containing movie metadata, genres, cast, crew, and user ratings.

**Dataset source:** [TMDb Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 🤖 How It Works

1. **Data Preprocessing**: Movie data is cleaned and processed
2. **Feature Engineering**: Text features are converted using TF-IDF
3. **Similarity Calculation**: Cosine similarity between movies is computed
4. **Recommendation Generation**: Top similar movies are selected
5. **UI Enhancement**: Posters and details are fetched via TMDb API

## 📈 Future Enhancements

- [ ] User rating system
- [ ] Collaborative filtering
- [ ] Deep learning recommendations
- [ ] User authentication
- [ ] Movie watchlist feature
- [ ] Advanced filtering options

## 📸 Screenshots

![App Screenshot](https://via.placeholder.com/800x400/4CAF50/white?text=Movie+Recommender+App)

*The app features a clean, intuitive interface with movie posters and detailed recommendations.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [The Movie Database (TMDb)](https://www.themoviedb.org/) for providing the movie data API
- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [Kaggle](https://www.kaggle.com/) for the dataset

---

# Made with ❤️ by Prince Thummar
**🔗 Live App:** [https://movie-recommender-system-by-prince.streamlit.app/](https://movie-recommender-system-by-prince.streamlit.app/)  
**📱 GitHub:** [https://github.com/PrinceThummar011/movie-recommender-system](https://github.com/PrinceThummar011/movie-recommender-system)  
**💼 LinkedIn:** [Connect with me](https://linkedin.com/in/prince-thummar)
