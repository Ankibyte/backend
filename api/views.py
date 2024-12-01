from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.conf import settings
from django.db import transaction
from django.utils import timezone
import os
import logging
from typing import Dict, Any, Optional
from .apkg_handler import ApkgHandler
from .models import ProcessingTask
from .services.visualization_service import VisualizationService
from .services.progress_service import ProcessingPhase
import uuid
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@api_view(['GET'])
def health_check(request):
    """Health check endpoint to verify API functionality."""
    return Response({
        "status": "ok",
        "message": "API is working!",
        "version": "1.0"
    })

@api_view(['GET'])
def upload_progress(request, task_id: str):
    """
    Get the progress of a processing task.
    
    Args:
        request: HTTP request object
        task_id: Unique identifier for the processing task
        
    Returns:
        Response containing task status and progress information
    """
    try:
        logger.info(f"Received progress request for task {task_id}")
        task = ProcessingTask.objects.get(task_id=task_id)
        
        # Prepare response data
        response_data = {
            'status': task.status,
            'progress_data': task.progress_data,
            'error_message': task.error_message
        }
        
        # Add output file URL if available
        if task.output_file:
            response_data['output_file'] = task.output_file.url
            
        logger.info(f"Progress data for task {task_id}: {task.progress_data}")
        return Response(response_data)
        
    except ProcessingTask.DoesNotExist:
        logger.warning(f"Task not found: {task_id}")
        return Response({
            'error': 'Task not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error checking progress: {str(e)}\n{traceback.format_exc()}")
        return Response({
            'error': 'Internal server error while checking progress'
        }, status=500)

@api_view(['GET'])
def process_results(request, task_id: str):
    """
    Get the final results of a completed processing task.
    
    Args:
        request: HTTP request object
        task_id: Unique identifier for the processing task
        
    Returns:
        Response containing task results including statistics and visualizations
    """
    try:
        logger.info(f"Fetching final results for task {task_id}")
        task = ProcessingTask.objects.get(task_id=task_id)
        
        if task.status != 'COMPLETED':
            return Response({
                'error': 'Task not completed yet'
            }, status=400)
            
        # Get handler to retrieve results
        handler = ApkgHandler()
        stats = handler.get_deck_statistics()
        
        # Get visualizations
        vis_service = VisualizationService()
        results = handler.get_processing_results()
        
        relevance_viz = vis_service.generate_relevance_distribution(results)
        embeddings = handler.get_embeddings()
        embedding_viz = None
        
        if embeddings is not None:
            embedding_viz = vis_service.generate_embedding_visualization(
                embeddings,
                [{
                    'relevance': r['tag'],
                    'similarity': r['similarity'],
                    'card_id': r['card_id']
                } for r in results]
            )
        
        response_data = {
            'status': task.status,
            'statistics': stats,
            'visualizations': {
                'relevance_distribution': relevance_viz,
                'embedding_visualization': embedding_viz
            },
            'processing_metrics': task.progress_data.get('processing_metrics', {}),
            'download_url': task.output_file.url if task.output_file else None
        }
        
        return Response(response_data)
        
    except ProcessingTask.DoesNotExist:
        logger.warning(f"Task not found: {task_id}")
        return Response({
            'error': 'Task not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Error fetching results: {str(e)}\n{traceback.format_exc()}")
        return Response({
            'error': 'Internal server error while fetching results'
        }, status=500)

@api_view(['POST'])
def process_deck(request):
    """Process an uploaded Anki deck file with study material."""
    task_id = None
    temp_files = []
    handler = None
    
    try:
        # Validate input files
        if 'anki_file' not in request.FILES:
            return Response({'error': 'No Anki file uploaded'}, status=400)
            
        if 'study_material' not in request.FILES:
            return Response({'error': 'No PDF study material uploaded'}, status=400)
            
        # Get request parameters
        anki_file = request.FILES['anki_file']
        pdf_file = request.FILES['study_material']
        custom_tag = request.data.get('custom_tag', '').strip()
        model_name = request.data.get('model', 'text-embedding-3-small')
        
        # Validate parameters
        if not anki_file.name.endswith('.apkg'):
            return Response({'error': 'Invalid Anki file type (must be .apkg)'}, status=400)
            
        if not pdf_file.name.endswith('.pdf'):
            return Response({'error': 'Invalid study material type (must be PDF)'}, status=400)
            
        if not custom_tag:
            return Response({'error': 'Tag prefix is required'}, status=400)
            
        # Create task
        task_id = str(uuid.uuid4())
        task = ProcessingTask.objects.create(
            task_id=task_id,
            status='PENDING',
            progress_data={
                'progress': 0,
                'phase': ProcessingPhase.INITIALIZING.value,
                'processed_cards': 0,
                'total_cards': 0,
                'estimated_time_remaining': None
            }
        )
        
        def update_progress(current_count: int, total_count: int) -> None:
            """Update task progress during card processing."""
            try:
                # Get current progress from handler
                if handler and handler.progress_tracker:
                    progress_data = handler.get_progress_data()
                    
                    # Update task with progress data
                    task.update_progress(progress_data)
                    
                    logger.info(
                        f"Progress: {progress_data['progress']:.1f}% - "
                        f"Phase: {progress_data['phase']} - "
                        f"({current_count}/{total_count} cards)"
                    )
                    
            except Exception as e:
                logger.error(f"Error updating progress: {str(e)}")
        
        # Initialize services
        vis_service = VisualizationService()
        
        # Setup directories
        tmp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
        processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save uploaded files
        anki_temp_path = os.path.join('tmp', anki_file.name)
        file_path = os.path.join(settings.MEDIA_ROOT, anki_temp_path)
        temp_files.append(file_path)
        
        with open(file_path, 'wb+') as destination:
            for chunk in anki_file.chunks():
                destination.write(chunk)
        
        try:
            # Update task status
            task.status = 'PROCESSING'
            task.save(update_fields=['status'])
            
            # Process deck
            with ApkgHandler(model_name=model_name, progress_callback=update_progress) as handler:
                # Extract deck
                deck_info = handler.extract_apkg(file_path)
                logger.info(f"Extracted deck info: {deck_info}")
                
                # Process with PDF
                results = handler.process_with_pdf(pdf_file, custom_tag)
                
                # Generate visualizations
                relevance_viz = vis_service.generate_relevance_distribution(results)
                embeddings = handler.get_embeddings()
                embedding_viz = None
                
                if embeddings is not None:
                    embedding_viz = vis_service.generate_embedding_visualization(
                        embeddings,
                        [{
                            'relevance': r['tag'],
                            'similarity': r['similarity'],
                            'card_id': r['card_id']
                        } for r in results]
                    )
                
                # Create output file
                output_filename = f"{os.path.splitext(anki_file.name)[0]}_tagged.apkg"
                output_path = os.path.join(settings.MEDIA_ROOT, 'processed', output_filename)
                
                handler.create_apkg(output_path)
                
                if not os.path.exists(output_path):
                    raise Exception("Failed to create processed deck file")
                
                # Update task completion
                task.output_file = os.path.join('processed', output_filename)
                task.status = 'COMPLETED'
                task.progress_data = handler.get_progress_data()
                task.save()
                
                # Get statistics
                stats = handler.get_deck_statistics()
                processing_time = (timezone.now() - task.created_at).total_seconds()
                
                # Prepare visualization data
                visualizations_data = {
                    'relevance_distribution': relevance_viz.to_dict() if relevance_viz else None,
                    'embedding_visualization': embedding_viz.to_dict() if embedding_viz else None,
                }
                
                # Return response
                return Response({
                    'status': 'success',
                    'message': 'Deck processed successfully',
                    'task_id': task_id,
                    'download_url': f"{request.build_absolute_uri('/')[:-1]}{task.output_file.url}",
                    'statistics': {
                        'total_cards': len(results),
                        'high_relevance': len([r for r in results if r['tag'] == 'high']),
                        'medium_relevance': len([r for r in results if r['tag'] == 'medium']),
                        'low_relevance': len([r for r in results if r['tag'] == 'low']),
                        'deck_info': stats
                    },
                    'visualizations': visualizations_data,
                    'processing_metrics': {
                        'embedding_model': model_name,
                        'processing_time': processing_time,
                        'pdf_pages': handler.pdf_service.metrics.total_pages,
                        'chunks_processed': handler.pdf_service.metrics.total_chunks
                    }
                })
                
        except Exception as e:
            logger.error(f"Error processing deck: {str(e)}\n{traceback.format_exc()}")
            if task_id:
                task.status = 'FAILED'
                task.error_message = str(e)
                task.save(update_fields=['status', 'error_message'])
            raise
            
    except Exception as e:
        error_message = f"Process deck error: {str(e)}"
        logger.error(f"{error_message}\n{traceback.format_exc()}")
        return Response({
            'error': error_message,
            'status': 'error'
        }, status=500)
        
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_file}: {str(e)}")