import os
import logging
import tempfile
import shutil
import time
import re
import zipfile
from pathlib import Path
import io
import numpy as np
from collections import Counter
from openai import OpenAI
from typing import Optional, List, Dict, Any, Tuple, Callable
import json
import sqlite3
from datetime import datetime
from .services.pdf_service import PDFService
from .services.visualization_service import VisualizationService, VisualizationData
from .services.progress_service import ProgressTracker, ProcessingPhase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s - %(name)s:%(lineno)d'
)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "text-embedding-3-small", progress_callback: Optional[Callable[[int, int], None]] = None):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_name = model_name
        self.embeddings_cache = {}
        self.progress_callback = progress_callback

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            text_hash = hash(text)
            if text_hash in self.embeddings_cache:
                return self.embeddings_cache[text_hash]

            text = text.replace("\n", " ").strip()
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text],
                encoding_format="float"
            )
            embedding = np.array(response.data[0].embedding)
            
            self.embeddings_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
        
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def get_relevance_tag(self, similarity_score: float) -> str:
        if similarity_score > 0.8:
            return "high"
        elif similarity_score > 0.5:
            return "medium"
        return "low"

    def process_cards(self, cards: List[Dict], pdf_content: str) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        try:
            pdf_vector = self.get_embedding(pdf_content)
            
            results = []
            card_embeddings = []
            total_cards = len(cards)
            
            for idx, card in enumerate(cards, 1):
                card_content = f"{card['front']}\n{card['back']}"
                card_vector = self.get_embedding(card_content)
                card_embeddings.append(card_vector)
                
                similarity = self.compute_similarity(card_vector, pdf_vector)
                tag = self.get_relevance_tag(similarity)
                
                results.append({
                    'card_id': card['id'],
                    'similarity': float(similarity),
                    'tag': tag
                })
                
                if self.progress_callback:
                    self.progress_callback(idx, total_cards)

            embeddings_matrix = np.vstack(card_embeddings)
            
            similarity_matrix = np.array([
                [self.compute_similarity(e1, e2) for e2 in card_embeddings]
                for e1 in card_embeddings
            ])

            return results, embeddings_matrix, similarity_matrix
            
        except Exception as e:
            logger.error(f"Error processing cards: {str(e)}")
            raise

    def clear_cache(self):
        self.embeddings_cache.clear()

class ApkgHandler:
    def __init__(self, model_name: str = "text-embedding-3-small", progress_callback: Optional[Callable[[int, int], None]] = None):
        self.temp_dir = None
        self.collection_path = None
        self.media_dir = None
        self.media_map_file = None
        self.deck_name = None
        self.media_map = {}
        self.progress_tracker = None
        self.embedding_service = EmbeddingService(model_name=model_name, progress_callback=self._progress_update)
        self.pdf_service = PDFService(cache_dir='media/pdf_cache')
        self.visualization_service = VisualizationService()
        self.external_progress_callback = progress_callback
        
        self.embeddings = None
        self.similarity_matrix = None
        self.processing_results = None
        self.visualizations = {
            'relevance_distribution': None,
            'embedding_visualization': None,
        }

    def _progress_update(self, current: int, total: int) -> None:
        """Internal progress update that uses both tracker and callback"""
        if self.progress_tracker:
            self.progress_tracker.update_progress(current)
        if self.external_progress_callback:
            self.external_progress_callback(current, total)

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.media_dir = os.path.join(self.temp_dir, "media")
        self.media_map_file = os.path.join(self.temp_dir, "media_map.json")
        os.makedirs(self.media_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.embeddings = None
        self.similarity_matrix = None
        self.processing_results = None
        self.visualizations = None
        self.embedding_service.clear_cache()
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def extract_apkg(self, apkg_path: str) -> Dict:
        try:
            if self.progress_tracker:
                self.progress_tracker.start_phase(ProcessingPhase.EXTRACTING_DECK)

            with zipfile.ZipFile(apkg_path, 'r') as zf:
                file_list = zf.namelist()
                
                if 'collection.anki21' in file_list:
                    self.collection_path = os.path.join(self.temp_dir, "collection.anki21")
                    collection_name = "collection.anki21"
                else:
                    self.collection_path = os.path.join(self.temp_dir, "collection.anki2")
                    collection_name = "collection.anki2"

                zf.extract(collection_name, self.temp_dir)
                self.media_map = self._process_media(zf)

                conn = sqlite3.connect(self.collection_path)
                try:
                    cur = conn.cursor()
                    
                    cur.execute("SELECT flds FROM notes LIMIT 1")
                    first_note = cur.fetchone()
                    if first_note and "Please update to the latest Anki version" in first_note[0]:
                        raise Exception("This deck was exported without 'Support older Anki versions' enabled")

                    cur.execute("SELECT decks FROM col")
                    decks_data = cur.fetchone()[0]
                    decks = json.loads(decks_data)
                    
                    for deck_id, deck in decks.items():
                        if deck['name'] != 'Default':
                            self.deck_name = deck['name']
                            break
                    
                    if not self.deck_name:
                        self.deck_name = list(decks.values())[0]['name']

                    cur.execute("SELECT COUNT(*) FROM notes")
                    note_count = cur.fetchone()[0]

                    logger.info(f"Extracted deck: {self.deck_name} with {note_count} notes")
                    
                    if self.progress_tracker:
                        self.progress_tracker.complete_phase(ProcessingPhase.EXTRACTING_DECK)

                    return {
                        'deck_name': self.deck_name,
                        'note_count': note_count,
                        'media_files': len(self.media_map)
                    }

                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Error extracting .apkg file: {str(e)}")
            if self.progress_tracker:
                self.progress_tracker.current_phase = ProcessingPhase.FAILED
            raise

    def process_with_pdf(self, pdf_file, tag_prefix: str) -> List[Dict]:
        try:
            # Initialize progress tracker
            self.progress_tracker = ProgressTracker(pdf_file.size)
            self.progress_tracker.start_phase(ProcessingPhase.INITIALIZING)
            
            logger.info("Processing cards with PDF for relevance tagging")
            
            # Process PDF
            self.progress_tracker.start_phase(ProcessingPhase.ANALYZING_PDF)
            pdf_content = self.pdf_service.get_embedding_content(pdf_file)
            self.progress_tracker.complete_phase(ProcessingPhase.ANALYZING_PDF)
            
            if not pdf_content.strip():
                raise ValueError("No content could be extracted from PDF")

            conn = sqlite3.connect(self.collection_path)
            try:
                # Extract cards
                self.progress_tracker.start_phase(ProcessingPhase.EXTRACTING_DECK)
                cards_to_process, note_card_map = self._extract_card_data(conn)
                self.progress_tracker.complete_phase(ProcessingPhase.EXTRACTING_DECK)
                
                if not cards_to_process:
                    raise Exception("No valid cards found to process")

                logger.info(f"Found {len(cards_to_process)} notes to process")
                
                # Process embeddings
                self.progress_tracker.start_phase(
                    ProcessingPhase.GENERATING_EMBEDDINGS,
                    total_items=len(cards_to_process)
                )
                results, self.embeddings, self.similarity_matrix = self.embedding_service.process_cards(
                    cards_to_process,
                    pdf_content
                )
                self.progress_tracker.complete_phase(ProcessingPhase.GENERATING_EMBEDDINGS)
                
                if not results:
                    raise Exception("No results generated from processing")

                # Update tags
                self.progress_tracker.start_phase(ProcessingPhase.UPDATING_TAGS)
                self._update_card_tags(conn, results, note_card_map, tag_prefix)
                self.progress_tracker.complete_phase(ProcessingPhase.UPDATING_TAGS)
                
                # Store results for later use
                self.processing_results = results
                
                return results

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error processing cards with PDF: {str(e)}")
            if self.progress_tracker:
                self.progress_tracker.mark_failed(str(e))
            raise

    def create_apkg(self, output_path: str) -> None:
        try:
            if self.progress_tracker:
                self.progress_tracker.start_phase(ProcessingPhase.FINALIZING)

            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                collection_name = os.path.basename(self.collection_path)
                zf.write(self.collection_path, collection_name)

                for idx, (internal_name, original_name) in enumerate(sorted(self.media_map.items())):
                    media_path = os.path.join(self.media_dir, internal_name)
                    if os.path.exists(media_path) and os.path.isfile(media_path):
                        zf.write(media_path, f'media/{idx}')
                        self.media_map[str(idx)] = original_name

                if self.media_map:
                    zf.writestr('media', json.dumps(self.media_map))

            logger.info(f"Created .apkg file at {output_path} with {len(self.media_map)} media files")
            
            if self.progress_tracker:
                self.progress_tracker.complete_phase(ProcessingPhase.FINALIZING)
            
        except Exception as e:
            logger.error(f"Error creating .apkg file: {str(e)}")
            if self.progress_tracker:
                self.progress_tracker.current_phase = ProcessingPhase.FAILED
            raise

    def get_deck_statistics(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.collection_path):
                return {}

            conn = sqlite3.connect(self.collection_path)
            try:
                cur = conn.cursor()
                
                cur.execute("SELECT COUNT(*) FROM notes")
                note_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM cards")
                card_count = cur.fetchone()[0]
                
                media_count = len(self.media_map)
                
                relevance_stats = {}
                if self.processing_results:
                    tag_counts = Counter(r['tag'] for r in self.processing_results)
                    avg_similarity = np.mean([r['similarity'] for r in self.processing_results])
                    relevance_stats = {
                        'high_relevance': tag_counts.get('high', 0),
                        'medium_relevance': tag_counts.get('medium', 0),
                        'low_relevance': tag_counts.get('low', 0),
                        'average_similarity': float(avg_similarity)
                    }
                
                return {
                    'total_notes': note_count,
                    'total_cards': card_count,
                    'deck_name': self.deck_name,
                    'media_files': media_count,
                    'processing_results': relevance_stats
                }
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error getting deck statistics: {str(e)}")
            return {}

    def get_progress_data(self) -> Dict[str, Any]:
        """Get current progress data."""
        if not self.progress_tracker:
            return {}
        
        progress_data = self.progress_tracker.get_progress_data()
        return {
            'progress': str(progress_data.progress * 100),  # Convert to percentage
            'phase': progress_data.phase.value,
            'processed_items': progress_data.processed_items,
            'total_items': progress_data.total_items,
            'estimated_time_remaining': progress_data.time_remaining,
            'error_count': progress_data.total_errors
        }

    # Helper methods remain the same
    def _process_media(self, zf: zipfile.ZipFile) -> Dict[str, str]:
        media_files = {}
        media_map = {}

        try:
            if 'media' in zf.namelist():
                with zf.open('media') as media_file:
                    content = media_file.read().decode('utf-8')
                    if content.strip():
                        media_map = json.loads(content)

            for filename in zf.namelist():
                if filename.startswith('media/') and not filename.endswith('/'):
                    zf.extract(filename, self.temp_dir)
                    base_name = os.path.basename(filename)
                    if base_name:
                        media_files[base_name] = os.path.join(self.media_dir, base_name)

            valid_media = {}
            for idx, (key, filename) in enumerate(sorted(media_map.items())):
                if filename not in media_files:
                    continue
                
                new_filename = str(idx)
                new_filepath = os.path.join(self.media_dir, new_filename)
                
                if os.path.exists(media_files[filename]):
                    shutil.move(media_files[filename], new_filepath)
                    valid_media[new_filename] = filename

            return valid_media

        except Exception as e:
            logger.error(f"Error processing media: {str(e)}")
            raise

    def _extract_card_data(self, conn) -> Tuple[List[Dict], Dict[int, List[int]]]:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT n.id, n.flds, n.tags, c.id as card_id
            FROM notes n
            JOIN cards c ON c.nid = n.id
        """)
        note_data = cur.fetchall()
        
        if not note_data:
            raise Exception("No notes found in deck")
        
        logger.info(f"Found {len(note_data)} notes to process")
        
        cards_to_process = []
        note_card_map = {}
        
        for note_id, fields, tags, card_id in note_data:
            fields_list = fields.split('\x1f')
            if len(fields_list) >= 2:
                front = self._clean_html(fields_list[0])
                back = self._clean_html(fields_list[1])
                
                if note_id not in note_card_map:
                    note_card_map[note_id] = []
                note_card_map[note_id].append(card_id)
                
                cards_to_process.append({
                    'id': str(note_id),
                    'front': front,
                    'back': back
                })
                
        return cards_to_process, note_card_map

    def _update_card_tags(self, conn, results: List[Dict], note_card_map: Dict[int, List[int]], tag_prefix: str) -> None:
        cur = conn.cursor()
        modified_count = 0
        
        for result in results:
            note_id = int(result['card_id'])
            tag = f"{tag_prefix}_{result['tag']}"
            
            cur.execute("SELECT tags FROM notes WHERE id = ?", (note_id,))
            current_tags = cur.fetchone()[0]
            
            tag_list = current_tags.strip().split() if current_tags else []
            tag_list = [t for t in tag_list if not t.startswith(tag_prefix)]
            tag_list.append(tag)
            new_tags = " ".join(tag_list)
            
            cur.execute(
                "UPDATE notes SET tags = ?, mod = ? WHERE id = ?",
                (new_tags, int(time.time() * 1000), note_id)
            )
            
            if note_id in note_card_map:
                for card_id in note_card_map[note_id]:
                    cur.execute(
                        "UPDATE cards SET mod = ? WHERE id = ?",
                        (int(time.time() * 1000), card_id)
                    )
            
            modified_count += 1

        conn.commit()
        logger.info(f"Successfully tagged {modified_count} notes/cards")

    @staticmethod
    def _clean_html(text: str) -> str:
        """Remove HTML tags from text while preserving content."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize whitespace (replace multiple spaces with single space)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_visualizations(self) -> Dict[str, Optional[VisualizationData]]:
        """Get all generated visualizations."""
        return self.visualizations

    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get the embeddings matrix if available."""
        return self.embeddings

    def get_similarity_matrix(self) -> Optional[np.ndarray]:
        """Get the similarity matrix if available."""
        return self.similarity_matrix

    def get_processing_results(self) -> Optional[List[Dict]]:
        """Get the processing results if available."""
        return self.processing_results

    def get_progress_data(self) -> Dict[str, Any]:
        """Get current progress information."""
        if self.progress_tracker:
            progress_data = self.progress_tracker.get_progress()
            return {
                'progress': progress_data.progress,
                'phase': progress_data.phase.value,
                'processed_items': progress_data.processed_items,
                'total_items': progress_data.total_items,
                'estimated_time_remaining': progress_data.time_remaining,
                'error_count': progress_data.error_count
            }
        return {
            'progress': 0,
            'phase': ProcessingPhase.INITIALIZING.value,
            'processed_items': 0,
            'total_items': 0,
            'estimated_time_remaining': None,
            'error_count': 0
        }