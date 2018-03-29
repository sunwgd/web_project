from django.db import models
from django.contrib import admin
# Create your models here.
class Images(models.Model):
    img=models.ImageField(upload_to='img')
