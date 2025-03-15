from django.urls import path
from .views import ProcessImageAPIView

urlpatterns = [
    path('process-image/', ProcessImageAPIView.as_view(), name='process_image_list'),
    path('process-image/<int:pk>/', ProcessImageAPIView.as_view(), name='process_image_detail'),
]
