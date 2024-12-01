from django.db import models
from django.utils import timezone
from django.db.models import F
from django.core.serializers.json import DjangoJSONEncoder

class ProcessingTask(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    
    task_id = models.CharField(max_length=255, unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    progress = models.IntegerField(default=0)
    progress_data = models.JSONField(
        null=True, 
        blank=True, 
        encoder=DjangoJSONEncoder,
        default=dict
    )
    error_message = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    output_file = models.FileField(upload_to='processed/', null=True, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['task_id']),
        ]
        verbose_name = 'Processing Task'
        verbose_name_plural = 'Processing Tasks'
    
    def update_progress(self, progress_data: dict) -> None:
        """
        Update progress data atomically.
        
        Args:
            progress_data: Dictionary containing updated progress information
        """
        self.__class__.objects.filter(pk=self.pk).update(
            progress_data=progress_data,
            updated_at=timezone.now()
        )
        
        # Refresh the instance after update
        self.refresh_from_db()
    
    def update_status(self, status: str, error_message: str = None) -> None:
        """
        Update task status atomically.
        
        Args:
            status: New status value
            error_message: Optional error message
        """
        if status not in dict(self.STATUS_CHOICES):
            raise ValueError(f"Invalid status: {status}")
            
        update_fields = {
            'status': status,
            'updated_at': timezone.now()
        }
        
        if error_message is not None:
            update_fields['error_message'] = error_message
            
        self.__class__.objects.filter(pk=self.pk).update(**update_fields)
        
        # Refresh the instance after update
        self.refresh_from_db()
    
    def get_duration(self) -> float:
        """
        Get task duration in seconds.
        
        Returns:
            Duration in seconds if task is completed, else time since creation
        """
        if self.status == 'COMPLETED':
            return (self.updated_at - self.created_at).total_seconds()
        return (timezone.now() - self.created_at).total_seconds()
    
    def is_active(self) -> bool:
        """
        Check if task is currently active.
        
        Returns:
            True if task is pending or processing, False otherwise
        """
        return self.status in ['PENDING', 'PROCESSING']
    
    def clean(self):
        """Validate model data."""
        from django.core.exceptions import ValidationError
        
        if self.status not in dict(self.STATUS_CHOICES):
            raise ValidationError({'status': f'Invalid status value: {self.status}'})
        
        if self.progress < 0 or self.progress > 100:
            raise ValidationError({'progress': 'Progress must be between 0 and 100'})
    
    def save(self, *args, **kwargs):
        """Override save to perform validation."""
        self.clean()
        super().save(*args, **kwargs)
    
    def __str__(self):
        """String representation of the model."""
        return f"Task {self.task_id} - {self.status}"