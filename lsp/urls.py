from django.urls import path
from .views import recognize_actions_from_video

urlpatterns = [
    path('recognize-actions-from-video/', recognize_actions_from_video, name='recognize-actions-from-video'),
]