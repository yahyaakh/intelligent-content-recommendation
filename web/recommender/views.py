
from django.shortcuts import render
from django.http import JsonResponse
from .engine import get_recommender

def index(request):
    return render(request, 'base.html')


# movie_id = int(input("Movie ID: "))
# n = int(input("Number of similar movies [5]: ") or 5)
# similar = self.find_similar_movies(movie_id, n)
# self.display_movies(similar, f"Movies Similar to ID {movie_id}")

def similar_movies(request):
    if request.method == 'POST':
        cli = get_recommender()

        # user can search by movie name or ID
        query = request.POST.get('query', '').strip()

        # try to resolve the query to a movie_id
        movie_id = None
        if query.isdigit():
            movie_id = int(query)
        else:
            matches = cli.movies_df[
                cli.movies_df['title'].str.contains(query, case=False, na=False)]
            if not matches.empty:
                movie_id = int(matches.iloc[0]['movie_id'])

        if movie_id is None or cli.movies_df[cli.movies_df['movie_id'] == movie_id].empty:
            return render(request, 'similar_movies.html', {
                'error': f'No movie found for "{query}".',
            })

        source = cli.movies_df[cli.movies_df['movie_id'] == movie_id].iloc[0].to_dict()
        source['genres_list'] = [g.strip() for g in str(source.get('genres', '')).split('|') if g.strip()]

        results = cli.find_similar_movies(movie_id, 10)
        for rec in results:
            rec['genres_list'] = [g.strip() for g in str(rec.get('genres', '')).split('|') if g.strip()]

        return render(request, 'similar_movies.html', {
            'source': source,
            'results': results,
        })

    return render(request, 'similar_movies.html')

def recommend(request):
    if request.method == 'POST':
        user_id   = int(request.POST.get('user_id') or 1)
        algorithm = request.POST.get('algorithm', 'hybrid')
        n = int(request.POST.get('recommendations') or 10)

        cli = get_recommender()

        # check if user exists
        if user_id not in cli.users_df['user_id'].values:
            return render(request, 'rec_list.html', {
                'error': f'User ID {user_id} does not exist.',
                'user_id': user_id,
                'algorithm': algorithm,
            })

        is_compare = request.POST.get('compare') == 'true'

        if is_compare:
            results = {}
            for algo in ['hybrid', 'collaborative', 'content', 'surprise', 'popular' , 'ncf']:
                recs = cli.get_user_recommendations(user_id, algo, 5)
                for rec in recs:
                    rec['genres_list'] = [g.strip() for g in str(rec.get('genres', '')).split('|') if g.strip()]
                    rec['cast_list'] = [c.strip() for c in str(rec.get('cast', '')).split('|') if c.strip()][:3]
                results[algo] = recs

            return render(request, 'rec_list.html', {
                'compare_results': results,
                'user_id': user_id,
                'is_compare': True,
            })
        else:
            recommendations = cli.get_user_recommendations(user_id, algorithm, n)
            top_movies = _get_top_movies(cli, user_id, 10)

            # build the user's taste profile once for explanations
            profile = cli.build_constraint_profile(user_id)

            for rec in recommendations:
                rec['genres_list'] = [g.strip() for g in str(rec.get('genres', '')).split('|') if g.strip()]
                rec['cast_list'] = [c.strip() for c in str(rec.get('cast', '')).split('|') if c.strip()][:3]
                rec['explanation'] = build_explanation(rec, profile)  # ← add this

            return render(request, 'rec_list.html', {
                'recommendations': recommendations,
                'user_id': user_id,
                'algorithm': algorithm,
                'top_movies': top_movies,
            })

    return render(request, 'rec_list.html')

def profile(request):
    if request.method == 'POST':
        user_id = int(request.POST.get('user_id') or 1)
        cli = get_recommender()

        if user_id not in cli.users_df['user_id'].values:
            return render(request, 'user_profile.html', {
                'error': f'User ID {user_id} does not exist.',
                'user_id': user_id,
            })

        profile = cli.get_user_profile(user_id)
        d = profile['demographics']
        s = profile['stats']
        t = profile['taste_profile']
        top_movies = _get_top_movies(cli, user_id, 10)

        # second check deleted

        return render(request, 'user_profile.html', {
            'user_id':    user_id,
            'age':        d['age'],
            'gender':     d['gender'],
            'occupation': d['occupation'],
            'age_group':  d['age_group'],
            'country':    d['country'],
            'total_movies':    s['total_movies'],
            'avg_rating':      s['avg_rating'],
            'avg_completion':  s['avg_completion'],
            'total_watch_time': s['total_watch_time'],
            'preferred_decade':   t['preferred_decade'],
            'preferred_language': t['preferred_language'],
            'avg_runtime':        t['avg_runtime_tolerance'],
            'liked_genres':       t['liked_genres'][:5],
            'liked_directors':    t['liked_directors'][:3],
            'liked_actors':       t['liked_actors'][:5],
            'top_movies':         top_movies,   # ← was missing
        })

    return render(request, 'user_profile.html')


def _get_top_movies(cli, user_id, n=10):
    interactions = cli.interactions_df
    movies = cli.movies_df

    user_ratings = interactions[interactions['user_id'] == user_id]
    top = user_ratings.nlargest(n, 'rating')

    # join with movie details
    merged = top.merge(movies, on='movie_id', how='left')

    results = []
    for _, row in merged.iterrows():
        genres = [g.strip() for g in str(row.get('genres', '')).split('|') if g.strip()]
        results.append({
            'title':  row.get('title', 'Unknown'),
            'year':   row.get('year', ''),
            'rating': row.get('rating', 0),
            'genres': genres[:3],
        })
    return results



def movie_search(request):
    """Autocomplete endpoint — returns matching movies as JSON."""
    query = request.GET.get('q', '').strip()
    cli = get_recommender()

    if len(query) < 2:
        return JsonResponse([], safe=False)

    matches = cli.movies_df[
        cli.movies_df['title'].str.contains(query, case=False, na=False)
    ].head(8)

    results = [
        {'movie_id': int(row['movie_id']),
         'title': row['title'],
         'year': int(row['year']) if str(row['year']).isdigit() else row['year']}
        for _, row in matches.iterrows()
    ]
    return JsonResponse(results, safe=False)


def personal_recommend(request):
    if request.method == 'POST':
        import json

        # the JS sends a JSON list of {movie_id, rating}
        data = json.loads(request.body)
        movie_ratings = data.get('movies', [])
        n = int(data.get('n', 10))

        if not movie_ratings:
            return JsonResponse({'error': 'No movies selected.'}, status=400)

        # the methods of the report (thesis engine), with constraints and the LLM options
        method = data.get('method', 'original')
        if method != 'original':
            return thesis_recommend(data, movie_ratings, method, n)

        cli = get_recommender()
        results = cli.get_recommendations_from_movies(movie_ratings, n)

        recs = []
        for rec in results:
            genres = [g.strip() for g in str(rec.get('genres', '')).split('|') if g.strip()]
            recs.append({
                'title':  rec.get('title', 'Unknown'),
                'year':   rec.get('year', ''),
                'genres': genres[:3],
                'director': rec.get('director', ''),
                'rating': round(float(rec.get('vote_average', 0)), 1),
            })

        return JsonResponse({'results': recs})

    return render(request, 'personal.html')




def build_explanation(movie, profile):
    reasons = []

    if movie.get('director') in profile.get('liked_directors', []):
        reasons.append(f"directed by {movie['director']}, one of your favourites")

    movie_cast = set(str(movie.get('cast', '')).split('|'))
    actor_overlap = movie_cast.intersection(profile.get('liked_actors', []))
    if actor_overlap:
        actors = ', '.join(list(actor_overlap)[:2])
        reasons.append(f"features {actors}")

    movie_genres = set(str(movie.get('genres', '')).split('|'))
    genre_overlap = movie_genres.intersection(profile.get('liked_genres', []))
    if genre_overlap:
        genres = ' & '.join(list(genre_overlap)[:2])
        reasons.append(f"matches your love of {genres}")

    movie_decade = (int(movie.get('year', 0)) // 10) * 10
    if movie_decade == profile.get('preferred_decade'):
        reasons.append(f"from the {movie_decade}s, your preferred decade")

    if not reasons:
        return "Recommended based on overall taste match"

    text = reasons[0]
    if len(reasons) > 1:
        text += " · " + " · ".join(reasons[1:])
    return text[0].upper() + text[1:]

# ───────────────────────── the methods of the report (thesis_engine.py) ─────────────────────────

def thesis_recommend(data, movie_ratings, method, n):
    """Popularity, Hybrid E, EASE or the learned ranker, then the hard constraints, then (optional)
    the LLM reorders the top 30. Returns the films, why films were removed, and time and tokens."""
    from .thesis_engine import get_engine, METHODS
    if method not in METHODS:
        return JsonResponse({'error': f'Unknown method {method!r}.'}, status=400)
    engine = get_engine()
    try:
        films, removed, info = engine.recommend(movie_ratings, method, n,
                                                request=data.get('constraints') or {},
                                                llm_rerank=bool(data.get('llm_rerank')))
    except ValueError as error:          # e.g. an unknown genre or no usable seed film
        return JsonResponse({'error': str(error)}, status=400)
    except Exception as error:           # e.g. the LLM could not be reached
        return JsonResponse({'error': f'{type(error).__name__}: {error}'}, status=500)

    recs = []
    for film in films:
        recs.append({
            'title': film.get('title', 'Unknown'),
            'year': int(film.get('year', 0)),
            'genres': [g for g in str(film.get('genres', '')).split('|') if g][:3],
            'director': film.get('director', ''),
            'rating': round(float(film.get('vote_average', 0)), 1),
            'audience': film.get('llm_audience', ''),
            'moods': [m for m in str(film.get('llm_mood', '')).split('|') if m],
        })
    return JsonResponse({'results': recs, 'info': info,
                         'removed': [{'title': t, 'reason': r} for t, r in removed[:15]]})


def thesis_options(request):
    """Genres, moods, themes and intensity dimensions for the constraint panel."""
    from .thesis_engine import get_engine
    return JsonResponse(get_engine().options())


def thesis_search(request):
    """Film search in the thesis catalog (the same 1,957 films and ids as the old pages)."""
    from .thesis_engine import get_engine
    query = request.GET.get('q', '').strip()
    if len(query) < 2:
        return JsonResponse([], safe=False)
    return JsonResponse(get_engine().search(query), safe=False)


def parse_request(request):
    """A typed request -> constraint settings, by the LLM (recomm.LLMRequestParser)."""
    import json
    from .thesis_engine import get_engine
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only.'}, status=405)
    text = json.loads(request.body).get('text', '').strip()
    if not text:
        return JsonResponse({'error': 'Type a request first.'}, status=400)
    try:
        constraints, info = get_engine().parse_request(text)
    except ValueError as error:          # e.g. a name that is not in the catalog
        return JsonResponse({'error': str(error)}, status=400)
    except Exception as error:           # e.g. Ollama is not running
        return JsonResponse({'error': f'{type(error).__name__}: {error}'}, status=500)
    return JsonResponse({'constraints': constraints, 'info': info})
