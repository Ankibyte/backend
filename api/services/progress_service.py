from typing import Dict, Optional, NamedTuple
from datetime import datetime
import time
import logging
from dataclasses import dataclass
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)

class ProcessingPhase(Enum):
    """Defines different phases of the processing pipeline."""
    INITIALIZING = "Initializing process"
    VALIDATING_FILES = "Validating input files"
    EXTRACTING_DECK = "Extracting Anki deck"
    ANALYZING_PDF = "Analyzing study material"
    GENERATING_EMBEDDINGS = "Generating embeddings"
    ANALYZING_RELEVANCE = "Analyzing card relevance"
    UPDATING_TAGS = "Updating card tags"
    CREATING_VISUALIZATIONS = "Creating visualizations"
    FINALIZING = "Finalizing processed deck"
    COMPLETED = "Process completed"
    FAILED = "Process failed"

@dataclass
class PhaseMetrics:
    """Stores metrics for a processing phase."""
    start_time: float
    end_time: Optional[float] = None
    items_processed: int = 0
    total_items: int = 0
    error_count: int = 0
    retries: int = 0

class ProgressData(NamedTuple):
    """Represents current progress state."""
    progress: float
    phase: ProcessingPhase
    processed_items: int
    total_items: int
    time_remaining: Optional[float]
    error_count: int

class ProgressTracker:
    """
    Tracks progress across different processing phases with thread-safe updates.
    """
    
    def __init__(self, file_size: int):
        """
        Initialize progress tracker.
        
        Args:
            file_size: Size of the input file in bytes
        """
        self.file_size = file_size
        self.start_time = time.time()
        self.current_phase: Optional[ProcessingPhase] = None
        self._lock = Lock()
        
        # Phase weights (must sum to 100)
        self.phase_weights: Dict[ProcessingPhase, float] = {
            ProcessingPhase.INITIALIZING: 5,
            ProcessingPhase.VALIDATING_FILES: 5,
            ProcessingPhase.EXTRACTING_DECK: 10,
            ProcessingPhase.ANALYZING_PDF: 15,
            ProcessingPhase.GENERATING_EMBEDDINGS: 35,
            ProcessingPhase.ANALYZING_RELEVANCE: 15,
            ProcessingPhase.UPDATING_TAGS: 10,
            ProcessingPhase.CREATING_VISUALIZATIONS: 3,
            ProcessingPhase.FINALIZING: 2
        }
        
        # Phase metrics tracking
        self.phase_metrics: Dict[ProcessingPhase, PhaseMetrics] = {}
        
        # Error tracking
        self.total_errors = 0
        self.error_threshold = 10
        
        # Performance metrics (MB per second)
        self.performance_metrics = {
            ProcessingPhase.ANALYZING_PDF: 2.0,
            ProcessingPhase.GENERATING_EMBEDDINGS: 1.0,
            ProcessingPhase.ANALYZING_RELEVANCE: 1.5
        }

    def start_phase(self, phase: ProcessingPhase, total_items: int = 0) -> None:
        """
        Start a new processing phase.
        
        Args:
            phase: The phase to start
            total_items: Expected number of items to process
        """
        with self._lock:
            if phase not in self.phase_weights:
                raise ValueError(f"Invalid phase: {phase}")
            
            self.current_phase = phase
            self.phase_metrics[phase] = PhaseMetrics(
                start_time=time.time(),
                total_items=total_items
            )
            
            logger.info(f"Starting phase: {phase.value} (Items: {total_items})")

    def update_progress(self, items_completed: int, errors: int = 0) -> None:
        """
        Update progress for current phase.
        
        Args:
            items_completed: Number of items processed
            errors: Number of errors encountered
        """
        with self._lock:
            if not self.current_phase or self.current_phase not in self.phase_metrics:
                return
            
            metrics = self.phase_metrics[self.current_phase]
            metrics.items_processed = items_completed
            metrics.error_count += errors
            self.total_errors += errors
            
            if metrics.total_items > 0:
                progress = (items_completed / metrics.total_items) * 100
                if progress % 10 < (progress - items_completed / metrics.total_items * 100):
                    logger.info(
                        f"Phase {self.current_phase.value}: "
                        f"{progress:.1f}% complete "
                        f"({items_completed}/{metrics.total_items})"
                    )

    def complete_phase(self, phase: ProcessingPhase) -> None:
        """
        Mark a phase as completed.
        
        Args:
            phase: Phase to mark as completed
        """
        with self._lock:
            if phase not in self.phase_metrics:
                return
            
            metrics = self.phase_metrics[phase]
            metrics.end_time = time.time()
            duration = metrics.end_time - metrics.start_time
            
            logger.info(
                f"Completed phase: {phase.value} "
                f"(Duration: {duration:.2f}s, "
                f"Errors: {metrics.error_count})"
            )

    def get_progress(self) -> ProgressData:
        """
        Get current progress information.
        
        Returns:
            ProgressData tuple containing current progress state
        """
        with self._lock:
            if not self.current_phase:
                return ProgressData(0, ProcessingPhase.INITIALIZING, 0, 0, None, 0)
            
            # Calculate completed progress
            completed_weight = sum(
                self.phase_weights[phase]
                for phase in self.phase_metrics
                if phase != self.current_phase
            )
            
            # Add progress of current phase
            if self.current_phase in self.phase_metrics:
                metrics = self.phase_metrics[self.current_phase]
                if metrics.total_items > 0:
                    phase_progress = metrics.items_processed / metrics.total_items
                else:
                    phase_progress = self._estimate_phase_progress()
                    
                completed_weight += self.phase_weights[self.current_phase] * phase_progress
            
            # Calculate time remaining
            time_remaining = self._estimate_remaining_time()
            
            # Get current items count
            current_metrics = self.phase_metrics.get(self.current_phase)
            processed_items = current_metrics.items_processed if current_metrics else 0
            total_items = current_metrics.total_items if current_metrics else 0
            
            return ProgressData(
                progress=round(completed_weight, 1),
                phase=self.current_phase,
                processed_items=processed_items,
                total_items=total_items,
                time_remaining=time_remaining,
                error_count=self.total_errors
            )

    def _estimate_phase_progress(self) -> float:
        """
        Estimate progress for phases without item counts.
        
        Returns:
            Estimated progress as a fraction (0-1)
        """
        if not self.current_phase or self.current_phase not in self.phase_metrics:
            return 0.0
            
        metrics = self.phase_metrics[self.current_phase]
        elapsed_time = time.time() - metrics.start_time
        
        if self.current_phase in self.performance_metrics:
            # Use performance metrics for estimation
            expected_duration = (
                self.file_size / (1024 * 1024) / 
                self.performance_metrics[self.current_phase]
            )
            return min(elapsed_time / max(expected_duration, 1.0), 1.0)
            
        return min(elapsed_time / 30.0, 1.0)  # Default to 30 seconds max

    def _estimate_remaining_time(self) -> Optional[float]:
        """
        Estimate remaining processing time.
        
        Returns:
            Estimated time remaining in seconds
        """
        if not self.current_phase:
            return None
            
        elapsed_time = time.time() - self.start_time
        completed_weight = sum(
            self.phase_weights[phase]
            for phase, metrics in self.phase_metrics.items()
            if metrics.end_time is not None
        )
        
        # Add progress of current phase
        if self.current_phase in self.phase_metrics:
            metrics = self.phase_metrics[self.current_phase]
            if metrics.total_items > 0:
                phase_progress = metrics.items_processed / metrics.total_items
                completed_weight += self.phase_weights[self.current_phase] * phase_progress
        
        if completed_weight == 0:
            # Use file size-based estimation
            return sum(
                self.file_size / (1024 * 1024) / rate
                for rate in self.performance_metrics.values()
            )
        
        # Calculate remaining time based on progress
        return (elapsed_time / completed_weight) * (100 - completed_weight)

    def get_phase_timing(self, phase: ProcessingPhase) -> Optional[float]:
        """
        Get the duration of a specific phase.
        
        Args:
            phase: Phase to get timing for
            
        Returns:
            Duration in seconds if phase is completed, None otherwise
        """
        metrics = self.phase_metrics.get(phase)
        if metrics and metrics.end_time:
            return metrics.end_time - metrics.start_time
        return None

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for completed phases.
        
        Returns:
            Dict containing performance metrics
        """
        metrics = {}
        for phase, phase_metrics in self.phase_metrics.items():
            if phase_metrics.end_time:
                duration = phase_metrics.end_time - phase_metrics.start_time
                if phase_metrics.total_items > 0:
                    metrics[phase.value] = phase_metrics.items_processed / duration
                elif self.file_size > 0:
                    metrics[phase.value] = (self.file_size / (1024 * 1024)) / duration
        return metrics

    def mark_failed(self, error_message: str) -> None:
        """Mark the current phase as failed with an error message."""
        with self._lock:
            self.current_phase = ProcessingPhase.FAILED
            logger.error(f"Task failed: {error_message}")
            self.total_errors += 1
