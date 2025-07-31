import pandas as pd
import numpy as np

# Load datasets
movies = pd.read_csv("C:\\Users\\ronin\\Desktop\\dataset\\TMDB\\tmdb_5000_movies.csv")
credits = pd.read_csv("C:\\Users\\ronin\\Desktop\\dataset\\TMDB\\tmdb_5000_credits.csv")

# Merge on 'id'
movies = movies.merge(credits, on='title')

# Select important columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Handle missing values
movies.dropna(inplace=True)

import ast

# Function to convert JSON-like strings into list of names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Get director name only
def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ''

# Apply transformations
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])  # Top 3 actors
movies['crew'] = movies['crew'].apply(get_director)

# Combine all into 'tags'
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['crew'] = movies['crew'].apply(lambda x: [x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Lowercase all text
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

new_df.head()
import pandas as pd
import ast  # For safely evaluating strings to objects

# Load the CSV files
movies = pd.read_csv("C:\\Users\\ronin\\Desktop\\dataset\\TMDB\\tmdb_5000_movies.csv")
credits = pd.read_csv("C:\\Users\\ronin\\Desktop\\dataset\\TMDB\\tmdb_5000_credits.csv")

# Merge the data on 'title'
movies = movies.merge(credits, on='title')

# Keep only the required columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop any rows with null values
movies.dropna(inplace=True)

# Convert the stringified lists into real Python lists using ast.literal_eval
def convert(obj):
    return [item['name'] for item in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Get the top 3 cast members
def get_top_cast(obj):
    cast_list = ast.literal_eval(obj)
    return [actor['name'] for actor in cast_list[:3]]

movies['cast'] = movies['cast'].apply(get_top_cast)

# Get the director’s name from crew
def get_director(obj):
    for person in ast.literal_eval(obj):
        if person['job'] == 'Director':
            return [person['name']]
    return []

movies['crew'] = movies['crew'].apply(get_director)

# Convert overview to list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine all the tags into a single list
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new DataFrame with just title and tags
new_df = movies[['movie_id', 'title', 'tags']]

# Convert tags list to a string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Lowercase all text
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

print(new_df.head(3))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize CountVectorizer (you can tweak max_features and stop_words later)
cv = CountVectorizer(max_features=5000, stop_words='english')

# Vectorize the tags
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute cosine similarity matrix
similarity = cosine_similarity(vectors)

print(similarity.shape)
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        print("Movie not found in database.")
        return

    # Get the index of the movie
    index = new_df[new_df['title'].str.lower() == movie].index[0]
    
    # Get similarity scores for that movie
    distances = list(enumerate(similarity[index]))
    
    # Sort the movies based on similarity score
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    print(f"\nRecommended movies similar to '{new_df.iloc[index].title}':")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
if __name__ == '__main__':
    print(recommend('Inception'))

