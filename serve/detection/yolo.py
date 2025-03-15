import os
from django.conf import settings
from PIL import Image
import numpy as np
from ultralytics import YOLO

MODEL_NAME = "yolov10m.pt"
model_instance = None

def download_model_if_not_exist():
    global model_instance
    if model_instance is None:
        model_instance = YOLO(MODEL_NAME)
    return True

def process_image_yolo10m(input_path, unique_name):
    """
    Обрабатывает изображение с помощью YOLO.
    Возвращает:
      - output_filename (тот же unique_name, чтобы имена совпадали)
      - detected_classes (список найденных классов)
      - detected_details (подробная информация: для каждого объекта – класс, confidence и bbox)
    """
    global model_instance
    if model_instance is None:
        raise Exception("Модель не инициализирована. Сначала вызовите download_model_if_not_exist().")

    results = model_instance(input_path)

    detected_classes = []
    detected_details = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            # Получаем имя класса
            class_idx = int(box.cls[0])
            class_name = model_instance.names[class_idx]
            detected_classes.append(class_name)
            # Получаем confidence и координаты bbox (если доступны)
            confidence = float(box.conf[0]) if hasattr(box, 'conf') else None
            bbox = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
            detected_details.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox
            })

    output_filename = unique_name
    output_path = os.path.join(settings.MEDIA_ROOT, 'output_img', output_filename)

    annotated_image = results[0].plot()
    img = Image.fromarray(annotated_image)
    img.save(output_path)

    return output_filename, detected_classes, detected_details
