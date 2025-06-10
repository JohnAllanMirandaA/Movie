from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

app = Flask(__name__)

# -------- Load and preprocess data once when app starts --------
print("Loading data...")

movies = pd.read_csv(r'C:\Movie_recommend\data\movies_metadata.csv', low_memory=False)
credits = pd.read_csv(r'C:\Movie_recommend\data\credits.csv')
keywords = pd.read_csv(r'C:\Movie_recommend\data\keywords.csv')
ratings = pd.read_csv(r'C:\Movie_recommend\data\ratings_small.csv')

movies = movies.drop_duplicates(subset='id')
credits['id'] = credits['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)
movies['id'] = movies['id'].astype(str)

movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('[]')
movies['cast'] = movies['cast'].fillna('[]')
movies['crew'] = movies['crew'].fillna('[]')
movies['keywords'] = movies['keywords'].fillna('[]')

def get_director(crew):
    for i in literal_eval(crew):
        if i['job'] == 'Director':
            return i['name']
    return ''

def get_list(data):
    return [i['name'] for i in literal_eval(data)][:3]

movies['director'] = movies['crew'].apply(get_director)
movies['cast'] = movies['cast'].apply(get_list)
movies['keywords'] = movies['keywords'].apply(get_list)
movies['genres'] = movies['genres'].apply(get_list)

movies['soup'] = movies['overview'] + ' ' + \
                 movies['director'] + ' ' + \
                 movies['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['keywords'].apply(lambda x: ' '.join(x))

tfidf = TfidfVectorizer(stop_words='english', dtype=np.float32)
tfidf_matrix = tfidf.fit_transform(movies['soup'])
print("TF-IDF matrix created with shape:", tfidf_matrix.shape)

movies = movies.reset_index(drop=True)
indices = pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()

reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd = SVD()
svd.fit(trainset)

def find_movie_title(title):
    title_lower = title.lower().strip()
    if title_lower in indices.index:
        return title_lower
    #partial_matches = [t for t in indices.index if title_lower in t]
    # ...existing code...
    partial_matches = [t for t in indices.index if isinstance(t, str) and title_lower in t]
# ...existing code...
    if partial_matches:
        return partial_matches[0]
    return None

def hybrid_recommendations(title, userId, top_n=10):
    title_lower = title.lower().strip()
    # 1. Find all partial and exact matches
    matched_titles = [t for t in movies['title'] if isinstance(t, str) and title_lower in t.lower()]
    matched_titles_set = set([t.lower() for t in matched_titles])

    # 2. Get indices for matched titles
    matched_indices = [i for i, t in enumerate(movies['title']) if isinstance(t, str) and title_lower in t.lower()]
    # ...rest of your function...
    # 3. Get similar movies using content-based filtering
    # Use the first matched movie as the reference for similarity
    if matched_indices:
        idx = matched_indices[0]
    else:
        return ["Movie not found"]

    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores_idx = cosine_similarities.argsort()[::-1]

    # 4. Exclude already matched titles from similar movies
    # 4. Exclude already matched titles from similar movies
    similar_indices = [
        i for i in sim_scores_idx
        if i < len(movies) and isinstance(movies.iloc[i]['title'], str) and movies.iloc[i]['title'].lower() not in matched_titles_set
    ]
    # 5. Combine matched titles and similar movies
    recommendations = list(matched_titles)  # Start with all matched titles

    # Now get the rest based on predicted rating
    filtered_movies = movies.iloc[similar_indices][['id', 'title']].copy()
    filtered_movies['id'] = pd.to_numeric(filtered_movies['id'], errors='coerce')

    scores = []
    for row in filtered_movies.itertuples():
        try:
            pred = svd.predict(userId, int(row.id)).est
        except Exception:
            pred = 0
        scores.append(pred)

    filtered_movies['pred_rating'] = scores
    # Only add enough to reach top_n total recommendations
    needed = max(0, top_n - len(recommendations))
    if needed > 0:
        extra = filtered_movies.sort_values('pred_rating', ascending=False).head(needed)['title'].tolist()
        recommendations.extend(extra)

    # If more than top_n due to matches, trim
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for title in recommendations:
        if title not in seen:
            unique_recommendations.append(title)
            seen.add(title)
        if len(unique_recommendations) == top_n:
            break

    return unique_recommendations

# ------------------ Flask routes ------------------
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    title = data.get('title', '')
    genre = data.get('genre', '')
    useName = data.get('useName', False)
    useGenre = data.get('useGenre', False)
    useImdb = data.get('useImdb', False)
    top_n = 10

    filtered_movies = movies.copy()

    # Filter by genre if selected
    if useGenre and genre:
        genre_lower = genre.lower()
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(
            lambda x: any(genre_lower in g.lower() for g in x) if isinstance(x, list) else False
        )]

    # Filter by name if selected
    if useName and title:
        title_lower = title.lower()
        filtered_movies = filtered_movies[filtered_movies['title'].apply(
            lambda t: isinstance(t, str) and title_lower in t.lower()
        )]

    # If IMDb filter is selected, sort by rating
    if useImdb:
        filtered_movies['vote_average'] = pd.to_numeric(filtered_movies['vote_average'], errors='coerce')
        filtered_movies = filtered_movies.sort_values('vote_average', ascending=False)

    # Remove duplicates and get top N
    recs = filtered_movies.drop_duplicates('title').head(top_n)['title'].tolist()
    if not recs:
        recs = ["Movie not found"]

    return jsonify({'recommendations': recs})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
