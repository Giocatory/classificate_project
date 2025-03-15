import os
import uuid
import json
import requests
from urllib.parse import urlparse
from datetime import datetime
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
    if not original_name or '.' not in original_name:
        original_name = 'image.jpg'
    return f"{uuid.uuid4().hex[:8]}_{original_name}"

@extend_schema_view(
    get=extend_schema(
        summary="Получить записи с фильтрацией, сортировкой и пагинацией",
        description=(
            "Возвращает список записей из истории обработки с возможностью фильтрации по ключевому слову (`q`),\n"
            "сортировки по указанному полю (`sort_by`) с порядком сортировки (`order`: asc/desc) и пагинации.\n\n"
            "Дополнительные query-параметры:\n"
            "- **q**: ключевое слово для поиска (по имени или обнаружённым классам).\n"
            "- **source_type**: фильтр по источнику ('url' или 'file').\n"
            "- **start_date** и **end_date**: даты в формате YYYY-MM-DD для фильтрации по дате обработки.\n"
            "- **sort_by**: поле сортировки (например, `image_name`, `datetime_input`, `shape`). По умолчанию `datetime_input`.\n"
            "- **order**: порядок сортировки (`asc` или `desc`). По умолчанию `desc`.\n"
            "- **page**: номер страницы. По умолчанию 1.\n"
            "- **page_size**: количество записей на страницу. По умолчанию 20."
        ),
        parameters=[
            OpenApiParameter(name="q", required=False, type=OpenApiTypes.STR, location=OpenApiParameter.QUERY, description="Ключевое слово для поиска"),
            OpenApiParameter(name="source_type", required=False, type=OpenApiTypes.STR, location=OpenApiParameter.QUERY, description="Фильтр по источнику ('url' или 'file')"),
            OpenApiParameter(name="start_date", required=False, type=OpenApiTypes.STR, location=OpenApiParameter.QUERY, description="Начальная дата (YYYY-MM-DD)"),
            OpenApiParameter(name="end_date", required=False, type=OpenApiTypes.STR, location=OpenApiParameter.QUERY, description="Конечная дата (YYYY-MM-DD)"),
            OpenApiParameter(name="sort_by", required=False, type=OpenApiTypes.STR, location=OpenApiParameter.QUERY, description="Поле сортировки (например, image_name, datetime_input, shape)"),
            OpenApiParameter(name="order", required=False, type=OpenApiTypes.STR, location=OpenApiParameter.QUERY, description="Порядок сортировки: asc или desc"),
            OpenApiParameter(name="page", required=False, type=OpenApiTypes.INT, location=OpenApiParameter.QUERY, description="Номер страницы"),
            OpenApiParameter(name="page_size", required=False, type=OpenApiTypes.INT, location=OpenApiParameter.QUERY, description="Количество записей на страницу"),
        ],
        responses={200: inline_serializer(
            name="ProcessImageListResponse",
            fields={
                "total_count": serializers.IntegerField(),
                "page": serializers.IntegerField(),
                "page_size": serializers.IntegerField(),
                "results": DetectionHistorySerializer(many=True),
            }
        )},
    ),
    post=extend_schema(
        summary="Обработка изображения через POST",
        description=(
            "Принимает изображение двумя способами:\n\n"
            "1) JSON с полем `image_url` – ссылка на изображение.\n"
            "2) multipart/form-data с полем `image` – загруженный файл.\n\n"
            "Обрабатывает изображение, сохраняет файлы и результаты в БД, возвращая пути к файлам и ID записи."
        ),
        request={
            "application/json": inline_serializer(
                name="ProcessImageJSONRequest",
                fields={
                    "image_url": serializers.URLField(required=True, help_text="Ссылка на изображение")
                }
            ),
            "multipart/form-data": inline_serializer(
                name="ProcessImageMultipartRequest",
                fields={
                    "image": serializers.ImageField(required=True, help_text="Файл изображения")
                }
            ),
        },
        responses={
            200: inline_serializer(
                name="ProcessImageSuccessResponse",
                fields={
                    "input_image": serializers.URLField(),
                    "output_image": serializers.URLField(),
                    "db_record_id": serializers.IntegerField(),
                }
            ),
            400: inline_serializer(
                name="ProcessImageErrorResponse",
                fields={"error": serializers.CharField()},
            ),
        },
    )
)
class ProcessImageAPIView(APIView):
    parser_classes = (JSONParser, MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        # Получаем query-параметры
        qs = DetectionHistory.objects.all()

        # Фильтрация по существующим параметрам
        source = request.query_params.get("source_type")
        start_date = request.query_params.get("start_date")
        end_date = request.query_params.get("end_date")
        if source:
            qs = qs.filter(source_type=source)
        if start_date:
            qs = qs.filter(datetime_input__gte=start_date)
        if end_date:
            qs = qs.filter(datetime_input__lte=end_date)

        # Фильтрация по ключевому слову (q) – ищем в image_name и classes_from_img
        keyword = request.query_params.get("q")
        if keyword:
            qs = qs.filter(Q(image_name__icontains=keyword) | Q(classes_from_img__icontains=keyword))

        # Сортировка: по умолчанию сортируем по datetime_input
        sort_by = request.query_params.get("sort_by", "datetime_input")
        order = request.query_params.get("order", "desc")
        if order.lower() == "desc":
            qs = qs.order_by("-" + sort_by)
        else:
            qs = qs.order_by(sort_by)

        # Пагинация: номер страницы и размер страницы
        try:
            page = int(request.query_params.get("page", 1))
            page_size = int(request.query_params.get("page_size", 20))
        except ValueError:
            return Response({"error": "Некорректные значения пагинации."}, status=status.HTTP_400_BAD_REQUEST)

        total_count = qs.count()
        offset = (page - 1) * page_size
        qs = qs[offset: offset + page_size]

        serializer = DetectionHistorySerializer(qs, many=True)
        data = {
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "results": serializer.data,
        }
        return Response(data, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        # Логика обработки POST (без изменений)
        image_url = request.data.get('image_url', None)
        uploaded_file = request.FILES.get('image', None)
        if image_url and uploaded_file:
            return Response(
                {"error": "Одновременно не может быть и image_url, и image-файл."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not image_url and not uploaded_file:
            return Response(
                {"error": "Не передано ни файла, ни ссылки."},
                status=status.HTTP_400_BAD_REQUEST
            )

        source = "url" if image_url else "file"

        if image_url:
            original_name = os.path.basename(urlparse(image_url).path)
            unique_name = generate_unique_filename(original_name)
            try:
                resp = requests.get(image_url)
                if resp.status_code != 200:
                    return Response(
                        {"error": "Не удалось скачать изображение по ссылке."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            except Exception as e:
                return Response(
                    {"error": f"Ошибка при скачивании: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
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
            return Response(
                {"error": f"Не удалось открыть изображение: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            yolo.download_model_if_not_exist()
        except Exception as e:
            return Response(
                {"error": f"Ошибка загрузки модели: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            output_filename, detected_classes, detected_details = yolo.process_image_yolo10m(input_path, unique_name)
        except Exception as e:
            return Response(
                {"error": f"Ошибка обработки изображения: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

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
            source_type=source
        )

        input_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, 'input_img', unique_name))
        output_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, 'output_img', output_filename))

        return Response(
            {
                "input_image": input_url,
                "output_image": output_url,
                "db_record_id": record.id
            },
            status=status.HTTP_200_OK
        )
