import os
from django.conf import settings
from shutil import copyfile
from ultralytics import YOLO  # если нужно для изображений; для видео можно использовать упрощённую логику

# Существующая модель для изображений (если требуется)
MODEL_NAME = "yolov10m.pt"
model_instance = None

def download_model_if_not_exist():
    global model_instance
    if model_instance is None:
        model_instance = YOLO(MODEL_NAME)
    return True

def process_image_yolo10m(input_path, unique_name):
    # ... существующая логика обработки изображения ...
    results = model_instance(input_path)
    detected_classes = []
    detected_details = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            class_idx = int(box.cls[0])
            class_name = model_instance.names[class_idx]
            detected_classes.append(class_name)
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
    from PIL import Image
    img = Image.fromarray(annotated_image)
    img.save(output_path)
    return output_filename, detected_classes, detected_details

def process_video_rutube(input_path, unique_name):
    """
    Демонстрационная функция обработки видео.
    Здесь можно добавить извлечение ключевых кадров и их обработку.
    Для простоты, функция просто копирует видео из input_video в output_video.
    """
    # Импортируем shutil для копирования файла
    from shutil import copyfile
    output_path = os.path.join(settings.MEDIA_ROOT, 'output_video', unique_name)
    copyfile(input_path, output_path)
    # Здесь можно добавить реальную обработку, например, извлечение кадров, запуск модели и т.д.
    detected_classes = []     # пока пусто
    detected_details = []     # пока пусто
    return unique_name, detected_classes, detected_details
