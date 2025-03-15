from rest_framework import serializers
from .models import DetectionHistory

class DetectionHistorySerializer(serializers.ModelSerializer):
    # Преобразуем строку JSON в объект
    detailed_results = serializers.JSONField()

    class Meta:
        model = DetectionHistory
        fields = '__all__'
