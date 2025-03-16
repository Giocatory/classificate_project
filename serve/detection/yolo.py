import os
import cv2
from django.conf import settings
from PIL import Image
from ultralytics import YOLO

MODEL_NAME = "yolov10m.pt"
model_instance = None

def download_model_if_not_exist():
    """
    Инициализирует модель YOLO через ultralytics.
    Если веса не скачаны, они будут загружены автоматически.
    """
    global model_instance
    if model_instance is None:
        model_instance = YOLO(MODEL_NAME)
    return True

def process_image_yolo10m(input_path, unique_name):
    """
    Обрабатывает одно изображение с помощью YOLO.
    Возвращает:
      - output_filename,
      - detected_classes (список найденных классов),
      - detected_details (список словарей с информацией о каждом найденном объекте)
    """
    global model_instance
    if model_instance is None:
        raise Exception("Модель не инициализирована. Сначала вызовите download_model_if_not_exist().")

    results = model_instance(input_path)
    detected_classes = []
    detected_details = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            class_idx = int(box.cls[0])
            class_name = model_instance.names[class_idx]
            confidence = float(box.conf[0]) if hasattr(box, 'conf') else None
            bbox = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
            detected_classes.append(class_name)
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

def process_video_yolo10m(input_video_path, unique_name):
    """
    Обрабатывает всё видео кадр за кадром:
      - Считывает каждый кадр.
      - Прогоняет YOLO, наносит bounding boxes.
      - Записывает аннотированные кадры в выходной видеофайл (MP4).
      - Собирает детальную информацию по каждому кадру.
    Возвращает:
      - output_filename: имя выходного видеофайла,
      - unique_classes: список уникальных обнаруженных классов,
      - all_detected_details: список словарей с данными для каждого обнаруженного объекта с указанием номера кадра.
    """
    global model_instance
    if model_instance is None:
        raise Exception("Модель не инициализирована. Сначала вызовите download_model_if_not_exist().")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Не удалось открыть входное видео.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = unique_name  # используем то же имя
    output_path = os.path.join(settings.MEDIA_ROOT, 'output_video', output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_detected_classes = []
    all_detected_details = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_instance(frame)
        annotated_frame = results[0].plot()
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_idx = int(box.cls[0])
                class_name = model_instance.names[class_idx]
                confidence = float(box.conf[0]) if hasattr(box, 'conf') else None
                bbox = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
                all_detected_classes.append(class_name)
                all_detected_details.append({
                    "frame": frame_idx,
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                })

        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()

    unique_classes = list(set(all_detected_classes))
    return output_filename, unique_classes, all_detected_details
