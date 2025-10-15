from django.contrib import admin
from .models import PredictionHistory, DiseaseInfo
# Register your models here.
@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['plant_type', 'predicted_disease', 'confidence', 'timestamp']
    list_filter = ['plant_type', 'predicted_disease', 'timestamp']
    search_fields = ['predicted_disease', 'plant_type']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Prediction Details', {
            'fields': ('image', 'predicted_disease', 'confidence', 'plant_type')
        }),
        ('Recommendations', {
            'fields': ('fertilizer_recommendation', 'treatment_recommendation')
        }),
        ('Metadata', {
            'fields': ('timestamp',)
        }),
    )

@admin.register(DiseaseInfo)
class DiseaseInfoAdmin(admin.ModelAdmin):
    list_display = ['disease_name']
    search_fields = ['disease_name', 'description']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('disease_name', 'description')
        }),
        ('Medical Details', {
            'fields': ('symptoms', 'causes')
        }),
        ('Treatment & Prevention', {
            'fields': ('treatment', 'prevention', 'fertilizer')
        }),
    )
