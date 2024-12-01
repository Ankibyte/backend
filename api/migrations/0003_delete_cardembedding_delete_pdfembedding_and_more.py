# Generated by Django 4.2 on 2024-11-29 21:32

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0002_cardembedding_pdfembedding_and_more"),
    ]

    operations = [
        migrations.DeleteModel(
            name="CardEmbedding",
        ),
        migrations.DeleteModel(
            name="PDFEmbedding",
        ),
        migrations.AddField(
            model_name="processingtask",
            name="progress_data",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
