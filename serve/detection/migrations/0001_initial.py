# Generated by Django 5.1.6 on 2025-03-08 20:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DetectionHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_name', models.CharField(help_text='Имя изображения (одинаковое для исходного и обработанного)', max_length=255)),
                ('datetime_input', models.DateTimeField(auto_now_add=True, help_text='Дата и время, когда запрос был обработан')),
                ('shape', models.CharField(help_text='Размер изображения в формате WxH', max_length=50)),
                ('classes_from_img', models.TextField(help_text='Список обнаруженных классов (через запятую)')),
                ('path', models.CharField(help_text='Путь к выходному изображению в MEDIA', max_length=255)),
                ('input_path', models.CharField(default='', help_text='Путь к исходному изображению в MEDIA', max_length=255)),
            ],
        ),
    ]
