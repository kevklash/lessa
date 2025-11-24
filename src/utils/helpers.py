"""
General helper functions for the Sign Language Translator application.
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

def create_directory_if_not_exists(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """Save data to JSON file."""
    try:
        create_directory_if_not_exists(os.path.dirname(filepath))
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON file {filepath}: {e}")
        return False

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {filepath}: {e}")
        return None

def normalize_landmarks(landmarks: List[List[float]]) -> List[List[float]]:
    """Normalize hand landmarks to 0-1 range."""
    if not landmarks:
        return landmarks
    
    landmarks_array = np.array(landmarks)
    
    # Get bounding box
    min_vals = np.min(landmarks_array, axis=0)
    max_vals = np.max(landmarks_array, axis=0)
    
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    
    # Normalize
    normalized = (landmarks_array - min_vals) / ranges
    
    return normalized.tolist()

def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def calculate_angle(point1: List[float], point2: List[float], point3: List[float]) -> float:
    """Calculate angle between three points (point2 is the vertex)."""
    # Vectors from point2 to point1 and point3
    v1 = np.array(point1) - np.array(point2)
    v2 = np.array(point3) - np.array(point2)
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Return angle in degrees
    return np.degrees(np.arccos(cos_angle))

def smooth_predictions(predictions: List[str], window_size: int = 5) -> str:
    """Smooth predictions using majority voting in a sliding window."""
    if len(predictions) < window_size:
        return predictions[-1] if predictions else ""
    
    # Get the last window_size predictions
    recent_predictions = predictions[-window_size:]
    
    # Count occurrences
    counts = {}
    for pred in recent_predictions:
        counts[pred] = counts.get(pred, 0) + 1
    
    # Return most common prediction
    return max(counts, key=counts.get)

def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage string."""
    return f"{confidence * 100:.1f}%"

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def setup_logging() -> None:
    """Setup basic logging configuration."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sign_language_translator.log'),
            logging.StreamHandler()
        ]
    )

class PerformanceTimer:
    """Simple performance timer for measuring execution time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = datetime.now() - self.start_time
            print(f"{self.name} took: {duration.total_seconds():.3f} seconds")

def print_system_info():
    """Print system information for debugging."""
    import platform
    import sys
    
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")
    print("="*30)