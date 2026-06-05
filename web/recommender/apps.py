from django.apps import AppConfig

class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'

    def ready(self):
        # pre-load the recommendation engine when Django starts
        from .engine import get_recommender
        get_recommender()