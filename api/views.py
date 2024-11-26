# api/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.conf import settings
import os
from .apkg_handler import ApkgHandler
from .models import ProcessingTask
import uuid
import logging

logger = logging.getLogger(__name__)

@api_view(['GET'])
def health_check(request):
    """Simple health check endpoint"""
    return Response({
        "status": "ok",
        "message": "API is working!"
    })

@api_view(['GET'])
def upload_progress(request, task_id):
    """Get the progress of a processing task"""
    try:
        task = ProcessingTask.objects.get(task_id=task_id)
        return Response({
            'status': task.status,
            'progress': task.progress,
            'error_message': task.error_message,
            'output_file': task.output_file.url if task.output_file else None,
        })
    except ProcessingTask.DoesNotExist:
        return Response({'error': 'Task not found'}, status=404)

@api_view(['POST'])
def process_deck(request):
    """Process an uploaded Anki deck file"""
    try:
        # Validate input
        if 'anki_file' not in request.FILES:
            return Response({'error': 'No file uploaded'}, status=400)
            
        anki_file = request.FILES['anki_file']
        custom_tag = request.data.get('custom_tag')
        
        if not anki_file.name.endswith('.apkg'):
            return Response({'error': 'Invalid file type'}, status=400)
            
        if not custom_tag:
            return Response({'error': 'No tag provided'}, status=400)
            
        # Create task entry
        task_id = str(uuid.uuid4())
        task = ProcessingTask.objects.create(task_id=task_id)
        
        # Create necessary directories
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'tmp'), exist_ok=True)
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'processed'), exist_ok=True)
        
        # Save uploaded file temporarily
        temp_path = default_storage.save(
            f'tmp/{anki_file.name}',
            anki_file
        )
        file_path = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        try:
            task.status = 'PROCESSING'
            task.progress = 10
            task.save()
            
            with ApkgHandler() as handler:
                # Extract and process the deck
                handler.extract_apkg(file_path)
                task.progress = 40
                task.save()
                
                handler.add_tag_to_notes(custom_tag)
                task.progress = 70
                task.save()
                
                # Create output filename
                output_filename = f"{os.path.splitext(anki_file.name)[0]}_tagged.apkg"
                output_path = os.path.join(settings.MEDIA_ROOT, 'processed', output_filename)
                
                # Create the new deck
                handler.create_apkg(output_path)
                
                # Update task with output file
                task.output_file = f'processed/{output_filename}'
                task.status = 'COMPLETED'
                task.progress = 100
                task.save()
                
                return Response({
                    'status': 'success',
                    'message': 'Deck processed successfully',
                    'task_id': task_id,
                    'download_url': f"/media/processed/{output_filename}"
                })
                
        except Exception as e:
            logger.error(f"Error processing deck: {str(e)}")
            task.status = 'FAILED'
            task.error_message = str(e)
            task.save()
            raise
            
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        logger.error(f"Process deck error: {str(e)}")
        return Response({
            'error': str(e)
        }, status=500)