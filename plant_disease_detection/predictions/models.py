from django.db import models
from django.utils import timezone
# Create your models here.


class PredictionHistory(models.Model):
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    predicted_disease = models.CharField(max_length=200)
    confidence = models.FloatField()
    plant_type = models.CharField(max_length=100)
    fertilizer_recommendation = models.TextField(blank=True)
    treatment_recommendation = models.TextField(blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Prediction Histories'

    def __str__(self):
        return f"{self.plant_type} - {self.predicted_disease} ({self.timestamp.strftime('%Y-%m-%d %H:%M')})"

class DiseaseInfo(models.Model):
    disease_name = models.CharField(max_length=200, unique=True)
    description = models.TextField()
    symptoms = models.TextField(default='', blank=True)
    causes = models.TextField(null=True, blank=True)
    prevention = models.TextField(null=True, blank=True)
    treatment = models.TextField(null=True, blank=True)
    fertilizer = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name_plural = 'Disease Information'

    def __str__(self):
        return self.disease_name
