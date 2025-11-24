"""
Configuration management for the Sign Language Translator application.
"""

import yaml
import os
from pathlib import Path

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration if file is not found."""
        return {
            'camera': {
                'device_id': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'mediapipe': {
                'max_num_hands': 2,
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.5,
                'model_complexity': 1
            },
            'app': {
                'confidence_threshold': 0.8,
                'smoothing_window': 5,
                'display_landmarks': True,
                'show_confidence': True
            }
        }
    
    def get(self, key, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_camera_config(self):
        """Get camera configuration."""
        return self.config.get('camera', {})
    
    def get_mediapipe_config(self):
        """Get MediaPipe configuration."""
        return self.config.get('mediapipe', {})
    
    def get_app_config(self):
        """Get application configuration."""
        return self.config.get('app', {})
    
    def get_gestures(self):
        """Get list of gestures to recognize."""
        return self.config.get('gestures', {})
    
    def get_paths(self):
        """Get path configurations."""
        return self.config.get('paths', {})

# Global configuration instance
config = Config()