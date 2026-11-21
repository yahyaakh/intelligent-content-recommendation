from django.urls import path
from . import views

urlpatterns = [
    path('',          views.index,     name='index'),
    path('recommend/', views.recommend, name='recommend'),
    path('profile/', views.profile, name='profile'),
path('similar/', views.similar_movies, name='similar_movies'),
path('personal/',       views.personal_recommend, name='personal'),
path('movie-search/',   views.movie_search,        name='movie_search'),
    # the methods of the report (thesis_engine.py)
    path('thesis-search/',   views.thesis_search,  name='thesis_search'),
    path('thesis-options/',  views.thesis_options, name='thesis_options'),
    path('parse-request/',   views.parse_request,  name='parse_request'),
]
