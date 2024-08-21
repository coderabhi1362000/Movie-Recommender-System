import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv(dataset_path)
movies = movies.sort_values(by='vote_count', ascending=False)
movies = movies[movies['original_language'] == 'en']
movies = movies[['id', 'title', 'overview', 'genres', 'keywords']]
movies.dropna(subset=['keywords', 'title'], axis=0, inplace=True)
movies = movies.fillna(' ')
movies['genres'] = movies['genres'].str.replace(' ', '')
movies = movies.head(20000)
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']
movies = movies[['id', 'title', 'tags']]
movies['tags'] = movies['tags'].str.replace(',', ' ')
movies['tags'] = movies['tags'].str.replace('  ', ' ')
movies['tags'] = movies['tags'].apply(lambda x: x.lower() if isinstance(x, str) else x)

ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

movies['tags'] = movies['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
vector = vector.astype(np.int16)

similarity = cosine_similarity(vector).astype(np.float16)

def recommend(movie):
    if movie not in movies['title'].values:
        return "Movie not found."
    
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommendations = [movies.iloc[i[0]].title for i in movies_list]
    return recommendations
