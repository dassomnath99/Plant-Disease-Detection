from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name='home'),
    path('api/predict/', views.predict_disease, name='predict'),
    path('api/history/', views.get_history, name='history'),
]

