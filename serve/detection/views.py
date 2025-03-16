import os
import uuid
import json
import requests
from urllib.parse import urlparse
from django.conf import settings
from django.db.models import Q
from django.shortcuts import get_object_or_404
from PIL import Image

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser

from drf_spectacular.utils import (
    extend_schema, extend_schema_view, inline_serializer, OpenApiParameter, OpenApiTypes
)

from .models import DetectionHistory
from .serializers import DetectionHistorySerializer
from . import yolo

def generate_unique_filename(original_name: str) -> str:
    """
    Генерирует уникальное имя файла на основе оригинального имени.
    Если имя не задано или не содержит расширения, используется 'file.jpg'.
    """
    if not original_name or '.' not in original_name:
        original_name = 'file.jpg'
    return f"{uuid.uuid4().hex[:8]}_{original_name}"


@extend_schema_view(
    post=extend_schema(
        summary="Обработка изображения (файл или ссылка)",
        description=(
            "**POST /api/process-image/**\n\n"
            "Принимает изображение двумя способами:\n"
            "- JSON с полем `image_url` – ссылка на изображение.\n"
            "- multipart/form-data с полем `image` – файл изображения.\n\n"
            "Скачивает/сохраняет изображение, обрабатывает YOLO, сохраняет результат в БД и возвращает JSON с URL-адресами и ID записи."
        ),
        request={
            "application/json": inline_serializer(
                name="ImageJSONRequest",
                fields={
                    "image_url": serializers.URLField(required=True, help_text="Ссылка на изображение")
                }
            ),
            "multipart/form-data": inline_serializer(
                name="ImageMultipartRequest",
                fields={
                    "image": serializers.ImageField(required=True, help_text="Файл изображения")
                }
            ),
        },
        responses={
            200: inline_serializer(
                name="ImageProcessSuccessResponse",
                fields={
                    "input_image": serializers.URLField(),
                    "output_image": serializers.URLField(),
                    "db_record_id": serializers.IntegerField(),
                }
            ),
            400: inline_serializer(
                name="ImageProcessErrorResponse",
                fields={"error": serializers.CharField()},
            ),
        },
    )
)
class ProcessImageAPIView(APIView):
    """
    Эндпоинт для обработки изображений.
    """
    parser_classes = (JSONParser, MultiPartParser, FormParser)

    def post(self, request, format=None):
        image_url = request.data.get('image_url')
        uploaded_file = request.FILES.get('image')

        if image_url and uploaded_file:
            return Response({"error": "Нельзя одновременно передавать image_url и файл."},
                            status=status.HTTP_400_BAD_REQUEST)
        if not image_url and not uploaded_file:
            return Response({"error": "Не передано ни ссылки, ни файла."},
                            status=status.HTTP_400_BAD_REQUEST)

        source_type = "url" if image_url else "file"

        if image_url:
            original_name = os.path.basename(urlparse(image_url).path)
            unique_name = generate_unique_filename(original_name)
            try:
                resp = requests.get(image_url)
                if resp.status_code != 200:
                    return Response({"error": "Не удалось скачать изображение по ссылке."},
                                    status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"error": f"Ошибка при скачивании: {str(e)}"},
                                status=status.HTTP_400_BAD_REQUEST)
            input_path = os.path.join(settings.MEDIA_ROOT, 'input_img', unique_name)
            with open(input_path, 'wb') as f:
                f.write(resp.content)
        else:
            original_name = uploaded_file.name
            unique_name = generate_unique_filename(original_name)
            input_path = os.path.join(settings.MEDIA_ROOT, 'input_img', unique_name)
            with open(input_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

        try:
            with Image.open(input_path) as pil_img:
                width, height = pil_img.size
            shape_str = f"{width}x{height}"
        except Exception as e:
            return Response({"error": f"Не удалось открыть изображение: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            yolo.download_model_if_not_exist()
        except Exception as e:
            return Response({"error": f"Ошибка загрузки модели: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            output_filename, detected_classes, detected_details = yolo.process_image_yolo10m(input_path, unique_name)
        except Exception as e:
            return Response({"error": f"Ошибка обработки изображения: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        classes_str = ", ".join(set(detected_classes)) if detected_classes else ""
        output_path_relative = os.path.join('output_img', output_filename)
        detailed_results_json = json.dumps(detected_details, ensure_ascii=False)

        record = DetectionHistory.objects.create(
            image_name=unique_name,
            shape=shape_str,
            classes_from_img=classes_str,
            detailed_results=detailed_results_json,
            path=output_path_relative,
            input_path=os.path.join('input_img', unique_name),
            source_type=source_type
        )

        input_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, 'input_img', unique_name))
        output_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, 'output_img', output_filename))

        return Response({
            "input_image": input_url,
            "output_image": output_url,
            "db_record_id": record.id
        }, status=status.HTTP_200_OK)


@extend_schema_view(
    post=extend_schema(
        summary="Обработка видео целиком (ссылка) с использованием yt-dlp",
        description=(
            "**POST /api/process-video/**\n\n"
            "Принимает JSON с полем `video_url` – ссылка на видео (например, с Rutube).\n"
            "Использует yt-dlp для загрузки видео (так как Rutube-страница не предоставляет прямой файл),\n"
            "затем обрабатывает всё видео кадр за кадром через YOLO, формируя аннотированный ролик.\n"
            "Сохраняет запись в БД и возвращает JSON с обнаруженными объектами и URL аннотированного видео."
        ),
        request=inline_serializer(
            name="VideoJSONRequest",
            fields={
                "video_url": serializers.URLField(required=True, help_text="Ссылка на видео (например, https://rutube.ru/video/...)")
            }
        ),
        responses={
            200: inline_serializer(
                name="VideoProcessSuccessResponse",
                fields={
                    "db_record_id": serializers.IntegerField(),
                    "found_classes": serializers.ListField(child=serializers.CharField()),
                    "detailed_results": serializers.JSONField(),
                    "output_video_url": serializers.URLField(),
                    "message": serializers.CharField(),
                }
            ),
            400: inline_serializer(
                name="VideoProcessErrorResponse",
                fields={"error": serializers.CharField()},
            ),
        },
    )
)
class ProcessVideoAPIView(APIView):
    """
    Эндпоинт для обработки видео целиком.
    """
    parser_classes = (JSONParser,)

    def post(self, request, format=None):
        video_url = request.data.get('video_url')
        if not video_url:
            return Response({"error": "Не передано поле video_url."},
                            status=status.HTTP_400_BAD_REQUEST)

        original_name = os.path.basename(urlparse(video_url).path)
        if not original_name:
            original_name = "video.mp4"
        unique_name = generate_unique_filename(original_name)

        # Используем yt-dlp для загрузки видео
        try:
            import yt_dlp
        except ImportError:
            return Response({"error": "Библиотека yt-dlp не установлена. Установите её: pip install yt-dlp."},
                            status=status.HTTP_400_BAD_REQUEST)

        input_video_path = os.path.join(settings.MEDIA_ROOT, 'input_video', unique_name)
        ydl_opts = {
            'outtmpl': input_video_path,
            'format': 'bestvideo+bestaudio/best',
            'quiet': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            return Response({"error": f"Ошибка при загрузке видео через yt-dlp: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        if not os.path.exists(input_video_path):
            return Response({"error": "Видео не было успешно загружено."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Инициализируем модель, если ещё не инициализирована
        try:
            yolo.download_model_if_not_exist()
        except Exception as e:
            return Response({"error": f"Ошибка загрузки модели: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        # Обрабатываем всё видео кадр за кадром
        try:
            output_filename, detected_classes, detected_details = yolo.process_video_yolo10m(input_video_path, unique_name)
        except Exception as e:
            return Response({"error": f"Ошибка обработки видео: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        classes_str = ", ".join(set(detected_classes)) if detected_classes else ""
        detailed_results_json = json.dumps(detected_details, ensure_ascii=False)
        output_path_relative = os.path.join('output_video', output_filename)

        record = DetectionHistory.objects.create(
            image_name=unique_name,
            shape="video",
            classes_from_img=classes_str,
            detailed_results=detailed_results_json,
            path=output_path_relative,
            input_path=os.path.join('input_video', unique_name),
            source_type="video"
        )

        output_video_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, 'output_video', output_filename))

        return Response({
            "db_record_id": record.id,
            "found_classes": detected_classes,
            "detailed_results": json.loads(detailed_results_json),
            "output_video_url": output_video_url,
            "message": "Видео успешно обработано кадр за кадром."
        }, status=status.HTTP_200_OK)
