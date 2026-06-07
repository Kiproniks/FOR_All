from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("books", "0009_booksentence_globallogicalthoughtblock_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="sentencethought",
            name="noise",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="sentencethought",
            name="skip_reason",
            field=models.TextField(blank=True),
        ),
        migrations.AddIndex(
            model_name="sentencethought",
            index=models.Index(fields=["book", "noise"], name="books_sente_book_id_noise_idx"),
        ),
    ]
