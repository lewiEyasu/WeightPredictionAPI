from django.urls import path
from . import views 

urlpatterns = [
    path('weight/', views.WeightPrediction),
]