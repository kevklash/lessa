"""
Hand detection using MediaPipe for sign language recognition.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ..utils.config import config

class HandDetector:
    """Hand detector using MediaPipe Hands solution."""
    
    def __init__(self):
        """Initialize MediaPipe hands detector."""
        mp_config = config.get_mediapipe_config()
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize the hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=mp_config.get('max_num_hands', 2),
            min_detection_confidence=mp_config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=mp_config.get('min_tracking_confidence', 0.5),
            model_complexity=mp_config.get('model_complexity', 1)
        )
        
        self.results = None
    
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect hands in the image and return annotated image with hand data.
        
        Args:
            image: Input BGR image from camera
            
        Returns:
            Tuple of (annotated_image, hands_data)
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Process the image
        self.results = self.hands.process(rgb_image)
        
        # Convert back to BGR for OpenCV
        rgb_image.flags.writeable = True
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        hands_data = []
        
        if self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # Get hand classification (Left/Right)
                hand_label = "Unknown"
                if self.results.multi_handedness:
                    hand_label = self.results.multi_handedness[hand_idx].classification[0].label
                
                # Extract landmark coordinates
                landmarks = self._extract_landmarks(hand_landmarks, image.shape)
                
                # Create hand data dictionary
                hand_data = {
                    'label': hand_label,
                    'landmarks': landmarks,
                    'confidence': self.results.multi_handedness[hand_idx].classification[0].score if self.results.multi_handedness else 0.0
                }
                
                hands_data.append(hand_data)
                
                # Draw landmarks on the image
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_image, hands_data
    
    def _extract_landmarks(self, hand_landmarks, image_shape: Tuple[int, int, int]) -> List[List[float]]:
        """
        Extract normalized landmark coordinates.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            List of [x, y, z] coordinates for each landmark
        """
        landmarks = []
        height, width = image_shape[:2]
        
        for landmark in hand_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates, then back to normalized
            x = landmark.x  # Already normalized (0-1)
            y = landmark.y  # Already normalized (0-1)
            z = landmark.z  # Depth relative to wrist
            
            landmarks.append([x, y, z])
        
        return landmarks
    
    def get_hand_landmarks(self, hand_data: Dict[str, Any]) -> np.ndarray:
        """Get landmarks as numpy array for easier processing."""
        return np.array(hand_data['landmarks'])
    
    def is_hand_detected(self) -> bool:
        """Check if any hands are detected in the last processed frame."""
        return self.results is not None and self.results.multi_hand_landmarks is not None
    
    def get_num_hands(self) -> int:
        """Get number of detected hands."""
        if self.results and self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
    
    def draw_info(self, image: np.ndarray, hands_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw additional information on the image.
        
        Args:
            image: Input image
            hands_data: List of hand data dictionaries
            
        Returns:
            Image with information overlay
        """
        info_image = image.copy()
        
        # Draw number of hands detected
        cv2.putText(info_image, f"Hands detected: {len(hands_data)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw hand information
        for i, hand_data in enumerate(hands_data):
            y_offset = 70 + i * 60
            
            # Hand label and confidence
            label = hand_data['label']
            confidence = hand_data['confidence']
            
            cv2.putText(info_image, f"Hand {i+1}: {label}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(info_image, f"Confidence: {confidence:.2f}", 
                       (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return info_image
    
    def close(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()

# Hand landmark indices for easy reference
HAND_LANDMARKS = {
    'WRIST': 0,
    'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
    'INDEX_FINGER_MCP': 5, 'INDEX_FINGER_PIP': 6, 'INDEX_FINGER_DIP': 7, 'INDEX_FINGER_TIP': 8,
    'MIDDLE_FINGER_MCP': 9, 'MIDDLE_FINGER_PIP': 10, 'MIDDLE_FINGER_DIP': 11, 'MIDDLE_FINGER_TIP': 12,
    'RING_FINGER_MCP': 13, 'RING_FINGER_PIP': 14, 'RING_FINGER_DIP': 15, 'RING_FINGER_TIP': 16,
    'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
}