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

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Defines available text chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC_BOUNDARY = "semantic_boundary"
    HYBRID = "hybrid"

@dataclass
class ProcessingMetrics:
    """Stores metrics about PDF processing operations."""
    total_pages: int
    processed_pages: int
    total_chunks: int
    average_chunk_size: int
    processing_time: float
    embedding_calls: int
    cache_hits: int

@dataclass
class SemanticChunk:
    """Represents a processed semantic chunk of text."""
    text: str
    page_number: int
    chunk_index: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

class PDFProcessor:
    """
    Advanced PDF processing system with semantic analysis capabilities.
    Handles text extraction, chunking, and embedding generation with caching.
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
        Initialize the PDF processor with specified configuration.
        
        Args:
            cache_dir: Directory for caching results
            chunk_strategy: Strategy for text chunking
            chunk_size: Target size for text chunks
            chunk_overlap: Percentage overlap between chunks
            embedding_model: OpenAI embedding model to use
            max_retries: Maximum retries for API calls
        """
        # Initialize basic configuration
        self.cache_dir = cache_dir
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0.0, min(1.0, chunk_overlap))
        self.embedding_model = embedding_model
        self.max_retries = max_retries

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Set up caching directory if specified
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize processing metrics
        self.metrics = ProcessingMetrics(
            total_pages=0,
            processed_pages=0,
            total_chunks=0,
            average_chunk_size=0,
            processing_time=0.0,
            embedding_calls=0,
            cache_hits=0
        )

    def process_pdf(self, pdf_file) -> Tuple[List[SemanticChunk], ProcessingMetrics]:
        """
        Process PDF file with semantic analysis and chunking.
        
        Args:
            pdf_file: File-like object containing PDF data
            
        Returns:
            Tuple containing:
                - List of processed semantic chunks
                - Processing metrics
                
        Raises:
            Exception: If processing fails
        """
        try:
            start_time = datetime.now()
            
            # Extract raw text with enhanced error handling
            raw_text = self._extract_text(pdf_file)
            
            # Create semantic chunks based on selected strategy
            chunks = self._create_chunks(raw_text)
            
            # Process chunks with embeddings
            processed_chunks = self._process_chunks(chunks)
            
            # Update processing metrics
            self.metrics.processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics.total_chunks = len(processed_chunks)
            self.metrics.average_chunk_size = int(np.mean([len(chunk.text) for chunk in processed_chunks]))
            
            return processed_chunks, self.metrics
            
        except Exception as e:
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
                        extracted_text[page_num] = text
                        self.metrics.processed_pages += 1
                    else:
                        logger.warning(f"Empty text on page {page_num}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {e}")
                    continue
            
            if not extracted_text:
                raise Exception("No text could be extracted from the PDF")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            raise

    def _create_chunks(self, page_text: Dict[int, str]) -> List[SemanticChunk]:
        """
        Create semantic chunks from extracted text using specified strategy.
        
        Args:
            page_text: Dict mapping page numbers to extracted text
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        chunk_index = 0
        
        for page_num, text in page_text.items():
            if self.chunk_strategy == ChunkingStrategy.FIXED_SIZE:
                page_chunks = self._fixed_size_chunking(text, self.chunk_size)
            elif self.chunk_strategy == ChunkingStrategy.SEMANTIC_BOUNDARY:
                page_chunks = self._semantic_boundary_chunking(text)
            else:  # HYBRID
                page_chunks = self._hybrid_chunking(text, self.chunk_size)
            
            # Create SemanticChunk objects
            for chunk_text in page_chunks:
                chunks.append(SemanticChunk(
                    text=chunk_text,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    metadata={
                        "strategy": self.chunk_strategy.value,
                        "length": len(chunk_text)
                    }
                ))
                chunk_index += 1
        
        return chunks

    def _fixed_size_chunking(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size for chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        overlap_size = int(chunk_size * self.chunk_overlap)
        
        start = 0
        while start < len(text):
            # Calculate end position with overlap
            end = start + chunk_size
            if end + overlap_size < len(text):
                # Look for nearest sentence boundary
                boundary = text.find('. ', end)
                if boundary != -1 and boundary - end < 100:
                    end = boundary + 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - overlap_size
        
        return chunks

    def _semantic_boundary_chunking(self, text: str) -> List[str]:
        """
        Split text at semantic boundaries (paragraphs, sections).
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n')]
        
        # Combine short paragraphs
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            if not paragraph:
                continue
                
            # If adding paragraph exceeds target size, start new chunk
            if current_length + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    def _hybrid_chunking(self, text: str, chunk_size: int) -> List[str]:
        """
        Use hybrid approach combining fixed-size and semantic boundaries.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size for chunks
            
        Returns:
            List of text chunks
        """
        # First split by semantic boundaries
        semantic_chunks = self._semantic_boundary_chunking(text)
        final_chunks = []
        
        # Further split large semantic chunks
        for chunk in semantic_chunks:
            if len(chunk) > chunk_size:
                final_chunks.extend(self._fixed_size_chunking(chunk, chunk_size))
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    def _process_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """
        Process chunks with embeddings and caching.
        
        Args:
            chunks: List of semantic chunks to process
            
        Returns:
            List of processed chunks with embeddings
        """
        processed_chunks = []
        
        for chunk in chunks:
            try:
                # Generate hash for chunk text
                chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
                
                # Check cache for existing embedding
                embedding = self._get_cached_embedding(chunk_hash)
                
                if embedding is not None:
                    self.metrics.cache_hits += 1
                else:
                    # Generate new embedding
                    embedding = self._generate_embedding(chunk.text)
                    self.metrics.embedding_calls += 1
                    
                    # Cache the embedding
                    self._cache_embedding(chunk_hash, embedding)
                
                # Update chunk with embedding
                chunk.embedding = embedding
                processed_chunks.append(chunk)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.chunk_index}: {e}")
                continue
        
        return processed_chunks

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using OpenAI API.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        retries = 0
        while retries < self.max_retries:
            try:
                text = text.replace("\n", " ").strip()
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=[text],
                    encoding_format="float"
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                retries += 1
                if retries == self.max_retries:
                    raise Exception(f"Failed to generate embedding after {self.max_retries} attempts: {e}")
                logger.warning(f"Embedding attempt {retries} failed: {e}")

    def _get_cached_embedding(self, content_hash: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding if available.
        
        Args:
            content_hash: Hash of content to look up
            
        Returns:
            Optional[np.ndarray]: Cached embedding if available
        """
        if not self.cache_dir:
            return None
            
        cache_path = Path(self.cache_dir) / f"{content_hash}.npy"
        if cache_path.exists():
            try:
                return np.load(str(cache_path))
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                return None
        return None

    def _cache_embedding(self, content_hash: str, embedding: np.ndarray) -> None:
        """
        Cache embedding for future use.
        
        Args:
            content_hash: Hash of content
            embedding: Embedding vector to cache
        """
        if not self.cache_dir:
            return
            
        cache_path = Path(self.cache_dir) / f"{content_hash}.npy"
        try:
            np.save(str(cache_path), embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about processing operations.
        
        Returns:
            Dict containing processing statistics
        """
        return {
            "total_pages": self.metrics.total_pages,
            "processed_pages": self.metrics.processed_pages,
            "total_chunks": self.metrics.total_chunks,
            "average_chunk_size": self.metrics.average_chunk_size,
            "processing_time": self.metrics.processing_time,
            "embedding_calls": self.metrics.embedding_calls,
            "cache_hits": self.metrics.cache_hits,
            "cache_hit_rate": (
                self.metrics.cache_hits /
                (self.metrics.embedding_calls + self.metrics.cache_hits)
                if (self.metrics.embedding_calls + self.metrics.cache_hits) > 0
                else 0
            )
        }