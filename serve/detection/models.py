from django.db import models

class DetectionHistory(models.Model):
    image_name = models.CharField(max_length=255, help_text="Имя изображения (одинаковое для input и output)")
    datetime_input = models.DateTimeField(auto_now_add=True, help_text="Дата и время обработки")
    shape = models.CharField(max_length=50, help_text="Размер изображения в формате WxH")
    classes_from_img = models.TextField(help_text="Обнаруженные классы (через запятую)")
    path = models.CharField(max_length=255, help_text="Путь к обработанному изображению в MEDIA")
    input_path = models.CharField(max_length=255, help_text="Путь к исходному изображению в MEDIA")
    source_type = models.CharField(
        max_length=10,
        default="",
        help_text="Источник изображения: 'url' или 'file'"
    )
    detailed_results = models.TextField(
        blank=True,
        default="",
        help_text="Детальная информация о классификации (JSON)"
    )

    def __str__(self):
        return f"{self.image_name} - {self.datetime_input}"
