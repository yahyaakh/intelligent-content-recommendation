from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

import requests as r
import pandas as pd
import json
import time


# TMDB uses IDs for genres, so we need to map them to text
genre_mapping = {28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
                     80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
                     14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
                     9648: 'Mystery', 10749: 'Romance', 878: 'Sci-Fi',
                     10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'}


page = 3

movies_list = []

for page in range(1, 50):
    #1 - 14 =  13 pages = ~250 movies

    # for page in range(1, 51):  # ~1000 movies

    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}&page={page}"

    response = r.get(url).json()
    for movie in response["results"]:

       time.sleep(0.25)

       movie_id = movie['id']

# 2. Fetch specific details (Runtime & Director) using the append_to_response trick
       detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits,keywords"
       details = r.get(detail_url).json()

# Find the director in the crew list
       crew = details.get('credits', {}).get('crew', [])
       director = next((member['name'] for member in crew if member['job'] == 'Director'), 'Unknown')
# get the cast data

       cast_list = details.get('credits', {}).get('cast', [])
       cast = '|'.join([member['name'] for member in cast_list[:3]])

# Map genre IDs to the first string genre (e.g., [28, 12] -> 'Action')
       genre_names = [genre_mapping.get(g_id, 'Unknown') for g_id in movie.get('genre_ids', [])]
       primary_genre = genre_names[0] if genre_names else 'Unknown'

       keywords_data = details.get('keywords', {}).get('keywords', [])
       keywords = '|'.join([k['name'] for k in keywords_data[:10]])


       movies_list.append({
           'movie_id': movie_id,
           'title': movie['title'],
           'genres': '|'.join(genre_names),  # ← all genres, not primary_genre
           'director': director,
           'cast': cast,  # ← new
           'year': movie['release_date'][:4],
           'runtime': details.get('runtime', 120),
           'vote_average': movie['vote_average'],  # ← renamed from 'rating'
           'vote_count': movie.get('vote_count', 0),  # ← new
           'original_language': movie['original_language'],
           'overview': movie.get('overview', ''),
           'keywords': keywords,  # ← new
       })


# make sure the dir exist
os.makedirs("data/raw", exist_ok=True)


with open("data/raw/tmdb_raw.json", "w", encoding="utf-8") as f:
    json.dump(movies_list, f, indent=2, ensure_ascii=False)

print(movies_list)
