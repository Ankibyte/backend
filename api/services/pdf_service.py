"""
PDF Service Implementation
A comprehensive service for PDF processing with semantic analysis and chunk optimization.
Includes caching, error handling, and multiple text chunking strategies.

Author: Claude
Date: 2024-11-27
Version: 1.0
"""

import os
import PyPDF2
import io
import logging
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Defines available text chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC_BOUNDARY = "semantic"
    HYBRID = "hybrid"

@dataclass
class ProcessingMetrics:
    """Tracks PDF processing metrics."""
    total_pages: int = 0
    processed_pages: int = 0
    total_chunks: int = 0
    average_chunk_size: int = 0
    processing_time: float = 0.0
    embedding_calls: int = 0
    cache_hits: int = 0
    error_count: int = 0

@dataclass
class TextChunk:
    """Represents a chunk of processed text."""
    text: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

class PDFService:
    """
    Service for processing PDF files with advanced text extraction and chunking capabilities.
    Includes caching, error handling, and semantic analysis features.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        chunk_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        chunk_size: int = 2000,
        chunk_overlap: float = 0.1,
        embedding_model: str = "text-embedding-3-small",
        max_retries: int = 3
    ):
        """
        Initialize the PDF service with specified configuration.
        
        Args:
            cache_dir: Directory for caching results
            chunk_strategy: Strategy for text chunking
            chunk_size: Target size for text chunks
            chunk_overlap: Percentage overlap between chunks (0.0 to 1.0)
            embedding_model: OpenAI embedding model identifier
            max_retries: Maximum retries for API calls
        """
        # Configuration
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0.0, min(1.0, chunk_overlap))
        self.embedding_model = embedding_model
        self.max_retries = max_retries

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Set up cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.embedding_cache = self.cache_dir / 'embeddings'
            self.text_cache = self.cache_dir / 'text'
            self.embedding_cache.mkdir(exist_ok=True)
            self.text_cache.mkdir(exist_ok=True)

        # Initialize metrics
        self.metrics = ProcessingMetrics()

    def process_pdf(self, pdf_file) -> Tuple[List[TextChunk], ProcessingMetrics]:
        """
        Process PDF file with text extraction and chunking.
        
        Args:
            pdf_file: File-like object containing PDF data
            
        Returns:
            Tuple containing:
                - List of processed text chunks
                - Processing metrics
                
        Raises:
            Exception: If processing fails
        """
        start_time = datetime.now()
        
        try:
            # Extract text with enhanced error handling
            extracted_text = self._extract_text(pdf_file)
            
            # Create chunks based on strategy
            chunks = self._create_chunks(extracted_text)
            
            # Update metrics
            self.metrics.total_chunks = len(chunks)
            self.metrics.average_chunk_size = int(np.mean([
                len(chunk.text) for chunk in chunks
            ])) if chunks else 0
            self.metrics.processing_time = (
                datetime.now() - start_time
            ).total_seconds()
            
            return chunks, self.metrics
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _extract_text(self, pdf_file) -> Dict[int, str]:
        """
        Extract text content from PDF with enhanced error handling.
        
        Args:
            pdf_file: File-like object containing PDF data
            
        Returns:
            Dict mapping page numbers to extracted text
            
        Raises:
            Exception: If text extraction fails
        """
        try:
            # Generate content hash for caching
            pdf_file.seek(0)
            content_hash = hashlib.sha256(pdf_file.read()).hexdigest()
            
            # Check cache
            if self.cache_dir:
                cache_path = self.text_cache / f"{content_hash}.json"
                if cache_path.exists():
                    with open(cache_path, 'r') as f:
                        self.metrics.cache_hits += 1
                        return json.load(f)
            
            # Reset file pointer and create PDF reader
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            
            # Update metrics
            self.metrics.total_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            extracted_text = {}
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text.strip():
                        # Process and clean text
                        text = self._clean_text(text)
                        extracted_text[page_num] = text
                        self.metrics.processed_pages += 1
                    else:
                        logger.warning(f"Empty text on page {page_num}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {e}")
                    self.metrics.error_count += 1
                    continue
            
            if not extracted_text:
                raise Exception("No text could be extracted from the PDF")
            
            # Cache extracted text
            if self.cache_dir:
                with open(cache_path, 'w') as f:
                    json.dump(extracted_text, f)
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and normalized text
        """
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        text = ''.join(char for char in text if char.isprintable())
        
        # Normalize line endings
        text = text.replace('\r', '\n')
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        return text.strip()

    def _create_chunks(self, page_text: Dict[int, str]) -> List[TextChunk]:
        """
        Create text chunks using specified strategy.
        
        Args:
            page_text: Dict mapping page numbers to extracted text
            
        Returns:
            List of text chunks
        """
        chunks = []
        chunk_index = 0
        
        for page_num, text in sorted(page_text.items()):
            if self.chunk_strategy == ChunkingStrategy.FIXED_SIZE:
                page_chunks = self._fixed_size_chunking(text)
            elif self.chunk_strategy == ChunkingStrategy.SEMANTIC_BOUNDARY:
                page_chunks = self._semantic_boundary_chunking(text)
            else:  # HYBRID
                page_chunks = self._hybrid_chunking(text)
            
            # Create TextChunk objects
            start_char = 0
            for chunk_text in page_chunks:
                chunks.append(TextChunk(
                    text=chunk_text,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        'strategy': self.chunk_strategy.value,
                        'length': len(chunk_text),
                        'page': page_num,
                        'index': chunk_index
                    }
                ))
                start_char += len(chunk_text)
                chunk_index += 1
        
        return chunks

    def _fixed_size_chunking(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position with overlap
            end = start + self.chunk_size
            if end + int(self.chunk_size * self.chunk_overlap) < text_length:
                # Look for nearest sentence boundary
                boundary = text.find('. ', end - 50, end + 50)
                if boundary != -1:
                    end = boundary + 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - int(self.chunk_size * self.chunk_overlap)
        
        return chunks

    def _semantic_boundary_chunking(self, text: str) -> List[str]:
        """
        Split text at semantic boundaries.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Split by paragraph boundaries
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_length = len(paragraph)
            
            # If adding paragraph exceeds target size
            if current_length + para_length > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Handle paragraphs larger than chunk size
                if para_length > self.chunk_size:
                    # Split into sentences
                    sentences = paragraph.split('. ')
                    current_sentence = []
                    sentence_length = 0
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence:
                            if sentence_length + len(sentence) > self.chunk_size:
                                if current_sentence:
                                    chunks.append('. '.join(current_sentence) + '.')
                                current_sentence = [sentence]
                                sentence_length = len(sentence)
                            else:
                                current_sentence.append(sentence)
                                sentence_length += len(sentence) + 2  # For '. '
                    
                    if current_sentence:
                        chunks.append('. '.join(current_sentence) + '.')
                else:
                    current_chunk = [paragraph]
                    current_length = para_length
            else:
                current_chunk.append(paragraph)
                current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    def _hybrid_chunking(self, text: str) -> List[str]:
        """
        Combine semantic and fixed-size chunking strategies.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # First split by semantic boundaries
        semantic_chunks = self._semantic_boundary_chunking(text)
        final_chunks = []
        
        # Process each semantic chunk
        for chunk in semantic_chunks:
            # If chunk is too large, apply fixed-size chunking
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self._fixed_size_chunking(chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    def get_embedding_content(self, pdf_file) -> str:
        """
        Get optimized content for embedding generation.
        
        Args:
            pdf_file: File-like object containing PDF data
            
        Returns:
            Optimized text content for embedding
            
        Raises:
            Exception: If processing fails
        """
        try:
            # Process PDF into chunks
            chunks, _ = self.process_pdf(pdf_file)
            
            # Combine chunks with metadata
            content_parts = []
            for chunk in chunks:
                # Add page and position metadata
                content_parts.append(f"[Page {chunk.page_number}, Pos {chunk.chunk_index}] {chunk.text}")
            
            # Join with clear separators
            return "\n---\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error getting embedding content: {str(e)}")
            raise

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about processing operations.
        
        Returns:
            Dict containing processing statistics
        """
        return {
            "total_pages": self.metrics.total_pages,
            "processed_pages": self.metrics.processed_pages,
            "success_rate": (
                self.metrics.processed_pages / self.metrics.total_pages
                if self.metrics.total_pages > 0 else 0
            ),
            "total_chunks": self.metrics.total_chunks,
            "average_chunk_size": self.metrics.average_chunk_size,
            "processing_time": self.metrics.processing_time,
            "cache_hits": self.metrics.cache_hits,
            "error_count": self.metrics.error_count
        }

    def clear_cache(self):
        """Clear all cached data."""
        if self.cache_dir:
            try:
                # Clear embedding cache
                for file in self.embedding_cache.glob('*.npy'):
                    file.unlink()
                
                # Clear text cache
                for file in self.text_cache.glob('*.json'):
                    file.unlink()
                    
                logger.info("Cache cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing cache: {str(e)}")
                raise