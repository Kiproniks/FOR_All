import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("books", "0010_sentencethought_noise_skip_reason"),
    ]

    operations = [
        migrations.AddField(
            model_name="globallogicalthoughtblock",
            name="is_merged",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="globallogicalthoughtblock",
            name="merged_into",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="merged_sources",
                to="books.globallogicalthoughtblock",
            ),
        ),
    ]
