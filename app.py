from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import random
app = Flask(__name__)


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
    
    partial_matches = [t for t in indices.index if isinstance(t, str) and title_lower in t]

    if partial_matches:
        return partial_matches[0]
    return None

def hybrid_recommendations(title, userId, top_n=10):
    title_lower = title.lower().strip()
    matched_titles = [t for t in movies['title'] if isinstance(t, str) and title_lower in t.lower()]
    matched_titles_set = set([t.lower() for t in matched_titles])

    matched_indices = [i for i, t in enumerate(movies['title']) if isinstance(t, str) and title_lower in t.lower()]
    
    if matched_indices:
        idx = matched_indices[0]
    else:
        return ["Movie not found"]

    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores_idx = cosine_similarities.argsort()[::-1]

    
    similar_indices = [
        i for i in sim_scores_idx
        if i < len(movies) and isinstance(movies.iloc[i]['title'], str) and movies.iloc[i]['title'].lower() not in matched_titles_set
    ]
    recommendations = list(matched_titles)  

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
    needed = max(0, top_n - len(recommendations))
    if needed > 0:
        extra = filtered_movies.sort_values('pred_rating', ascending=False).head(needed)['title'].tolist()
        recommendations.extend(extra)

    seen = set()
    unique_recommendations = []
    for title in recommendations:
        if title not in seen:
            unique_recommendations.append(title)
            seen.add(title)
        if len(unique_recommendations) == top_n:
            break

    return unique_recommendations

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/genres', methods=['GET'])
def get_genres():
    all_genres = set()
    for genre_list in movies['genres']:
        if isinstance(genre_list, list):
            all_genres.update(genre_list)
        elif isinstance(genre_list, str):
            all_genres.update([g.strip() for g in genre_list.split(',') if g.strip()])
    return jsonify(sorted(all_genres))
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    title = data.get('title', '')
    genre = data.get('genre', '')
    useName = data.get('useName', False)
    useGenre = data.get('useGenre', False)
    useImdb = data.get('useImdb', False)
    year_from = data.get('yearFrom', '').strip()
    year_to = data.get('yearTo', '').strip()
    top_n = 10

    filtered_movies = movies.copy()
    filtered_movies = movies.copy()

    if year_from or year_to:
        filtered_movies = filtered_movies[pd.notnull(filtered_movies['release_date'])]
        filtered_movies = filtered_movies[filtered_movies['release_date'].astype(str).str.len() >= 4]
        filtered_movies['year'] = filtered_movies['release_date'].astype(str).str[:4].astype(int)
        if year_from and year_to:
            filtered_movies = filtered_movies[
                (filtered_movies['year'] >= int(year_from)) & (filtered_movies['year'] <= int(year_to))
            ]
        elif year_from:
            filtered_movies = filtered_movies[filtered_movies['year'] >= int(year_from)]
        elif year_to:
            filtered_movies = filtered_movies[filtered_movies['year'] <= int(year_to)]
    if useGenre and genre:
        genre_lower = genre.lower()
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(
            lambda x: genre_lower in str(x).lower()
        )]
        filtered_movies = filtered_movies.sample(frac=1, random_state=random.randint(0, 100000))

    if useName and title:
        title_lower = title.lower()
        filtered_movies = filtered_movies[filtered_movies['title'].apply(
            lambda t: isinstance(t, str) and title_lower in t.lower()
        )]

    if useImdb:
        filtered_movies['vote_average'] = pd.to_numeric(filtered_movies['vote_average'], errors='coerce')
        filtered_movies = filtered_movies.sort_values('vote_average', ascending=False)
    mood = data.get('mood', '').lower().strip()
    mood_keywords = {
        'happy': ['friendship', 'fun', 'family', 'comedy', 'joy', 'holiday'],
        'sad': ['tragedy', 'death', 'loss', 'drama', 'tearjerker'],
        'thrilling': ['thriller', 'suspense', 'chase', 'crime', 'mystery'],
        'romantic': ['romance', 'love', 'relationship', 'kiss'],
        'adventurous': ['adventure', 'journey', 'exploration', 'quest'],
        'scary': ['horror', 'ghost', 'killer', 'monster', 'fear'],
        'inspiring': ['biography', 'overcome', 'success', 'inspire', 'dream']
    }
    if mood and mood in mood_keywords:
        keywords_set = set(mood_keywords[mood])
        def has_mood(row):
            genres = row['genres'] if isinstance(row['genres'], list) else []
            keywords = row['keywords'] if isinstance(row['keywords'], list) else []
            return bool(keywords_set.intersection(set([g.lower() for g in genres + keywords])))
        filtered_movies = filtered_movies[filtered_movies.apply(has_mood, axis=1)]
    recs = filtered_movies.drop_duplicates('title').head(top_n)
    if recs.empty:
        return jsonify({'recommendations': []})

    result = []
    for _, row in recs.iterrows():
        result.append({
            'title': row['title'],
            'year': str(row['release_date'])[:4] if pd.notnull(row.get('release_date', None)) else '',
            'rating': row.get('vote_average', ''),
            'imdb_id': row.get('imdb_id', ''),

            'genres': ', '.join(row.get('genres', [])) if isinstance(row.get('genres', ''), list) else row.get('genres', ''),
            'overview': row.get('overview', ''),
            'director': row.get('director', ''),
            'cast': ', '.join(row.get('cast', [])) if isinstance(row.get('cast', ''), list) else row.get('cast', ''),
            'keywords': ', '.join(row.get('keywords', [])) if isinstance(row.get('keywords', ''), list) else row.get('keywords', '')        })

    return jsonify({'recommendations': result})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
