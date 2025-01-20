# Generated by Django 4.1 on 2024-12-14 12:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('products', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='product',
            old_name='price',
            new_name='max_price',
        ),
        migrations.AddField(
            model_name='product',
            name='min_price',
            field=models.DecimalField(decimal_places=2, default=900, max_digits=10),
            preserve_default=False,
        ),
    ]
