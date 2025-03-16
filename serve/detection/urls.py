from django.urls import path
from .views import ProcessImageAPIView, ProcessVideoAPIView

urlpatterns = [
    path('process-image/', ProcessImageAPIView.as_view(), name='process_image'),
    path('process-video/', ProcessVideoAPIView.as_view(), name='process_video'),
]
