"""
Camera utilities for webcam handling and video processing.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from .config import config

class Camera:
    """Camera handler for webcam operations."""
    
    def __init__(self, device_id: Optional[int] = None):
        """Initialize camera with specified device ID."""
        camera_config = config.get_camera_config()
        
        self.device_id = device_id if device_id is not None else camera_config.get('device_id', 0)
        self.width = camera_config.get('width', 640)
        self.height = camera_config.get('height', 480)
        self.fps = camera_config.get('fps', 30)
        
        self.camera = None
        self.is_opened = False
    
    def start(self) -> bool:
        """Start the camera capture."""
        try:
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                print(f"Error: Could not open camera with device ID {self.device_id}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_opened = True
            print(f"Camera started successfully: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if not self.is_opened or self.camera is None:
            return False, None
        
        ret, frame = self.camera.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
        
        return ret, frame
    
    def stop(self):
        """Stop the camera and release resources."""
        if self.camera is not None:
            self.camera.release()
            self.is_opened = False
            print("Camera stopped and resources released")
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get current frame dimensions."""
        return self.width, self.height
    
    def is_camera_available(self) -> bool:
        """Check if camera is available and working."""
        test_camera = cv2.VideoCapture(self.device_id)
        if test_camera.isOpened():
            ret, _ = test_camera.read()
            test_camera.release()
            return ret
        return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

def list_available_cameras() -> list:
    """List all available camera devices."""
    available_cameras = []
    
    # Check first 10 device IDs
    for i in range(10):
        test_camera = cv2.VideoCapture(i)
        if test_camera.isOpened():
            ret, _ = test_camera.read()
            if ret:
                available_cameras.append(i)
        test_camera.release()
    
    return available_cameras

def test_camera(device_id: int = 0) -> bool:
    """Test if a specific camera device works."""
    try:
        with Camera(device_id) as cam:
            ret, frame = cam.read_frame()
            return ret and frame is not None
    except Exception:
        return False