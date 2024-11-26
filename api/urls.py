# api/urls.py
from django.urls import path
from .views import health_check, process_deck, upload_progress

urlpatterns = [
    path('health/', health_check, name='health_check'),
    path('process-deck/', process_deck, name='process_deck'),
    path('upload-progress/<str:task_id>/', upload_progress, name='upload_progress'),
]