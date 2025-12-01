"""
Holistic detection using MediaPipe for comprehensive sign language recognition.
Supports hands, body pose, and face detection for LESSA (El Salvador Sign Language System).
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ..utils.config import config

class HolisticDetector:
    """Holistic detector using MediaPipe Holistic solution for complete body tracking."""
    
    def __init__(self):
        """Initialize MediaPipe holistic detector."""
        mp_config = config.get_mediapipe_config()
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize the holistic detector
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=mp_config.get('model_complexity', 1),
            smooth_landmarks=True,
            enable_segmentation=False,  # Set to True if you want background segmentation
            smooth_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=mp_config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=mp_config.get('min_tracking_confidence', 0.5)
        )
        
        self.results = None
    
    def detect_holistic(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect hands, body pose, and face in the image.
        
        Args:
            image: Input BGR image from camera
            
        Returns:
            Tuple of (annotated_image, detection_data)
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Process the image
        self.results = self.holistic.process(rgb_image)
        
        # Convert back to BGR for OpenCV
        rgb_image.flags.writeable = True
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Extract all detection data
        detection_data = self._extract_detection_data(image.shape)
        
        # Draw all landmarks
        annotated_image = self._draw_landmarks(annotated_image)
        
        return annotated_image, detection_data
    
    def _extract_detection_data(self, image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Extract all detection data from MediaPipe results.
        
        Args:
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            Dictionary containing all detected landmarks and metadata
        """
        height, width = image_shape[:2]
        
        detection_data = {
            'timestamp': None,  # Will be set by caller
            'image_dimensions': {'width': width, 'height': height},
            'hands': {
                'left_hand': None,
                'right_hand': None
            },
            'pose': None,
            'face': None,
            'detection_confidence': {}
        }
        
        # Extract hand landmarks
        if self.results.left_hand_landmarks:
            detection_data['hands']['left_hand'] = self._extract_landmarks(
                self.results.left_hand_landmarks, 'left_hand'
            )
        
        if self.results.right_hand_landmarks:
            detection_data['hands']['right_hand'] = self._extract_landmarks(
                self.results.right_hand_landmarks, 'right_hand'
            )
        
        # Extract pose landmarks
        if self.results.pose_landmarks:
            detection_data['pose'] = self._extract_landmarks(
                self.results.pose_landmarks, 'pose'
            )
        
        # Extract face landmarks
        if self.results.face_landmarks:
            detection_data['face'] = self._extract_landmarks(
                self.results.face_landmarks, 'face'
            )
        
        # Calculate detection confidence scores
        detection_data['detection_confidence'] = {
            'left_hand': 1.0 if self.results.left_hand_landmarks else 0.0,
            'right_hand': 1.0 if self.results.right_hand_landmarks else 0.0,
            'pose': 1.0 if self.results.pose_landmarks else 0.0,
            'face': 1.0 if self.results.face_landmarks else 0.0
        }
        
        return detection_data
    
    def _extract_landmarks(self, landmarks, landmark_type: str) -> Dict[str, Any]:
        """
        Extract normalized landmark coordinates.
        
        Args:
            landmarks: MediaPipe landmarks object
            landmark_type: Type of landmarks ('left_hand', 'right_hand', 'pose', 'face')
            
        Returns:
            Dictionary with landmark data and metadata
        """
        landmark_data = {
            'type': landmark_type,
            'landmarks': [],
            'landmark_count': len(landmarks.landmark),
            'visibility_scores': []
        }
        
        for landmark in landmarks.landmark:
            # Extract coordinates
            x = landmark.x  # Normalized (0-1)
            y = landmark.y  # Normalized (0-1)
            z = landmark.z  # Depth (for hands) or relative depth (for pose/face)
            
            landmark_coords = [x, y, z]
            
            # Add visibility score if available (pose landmarks have visibility)
            if hasattr(landmark, 'visibility'):
                visibility = landmark.visibility
                landmark_coords.append(visibility)
                landmark_data['visibility_scores'].append(visibility)
            
            landmark_data['landmarks'].append(landmark_coords)
        
        return landmark_data
    
    def _draw_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Draw all detected landmarks on the image.
        
        Args:
            image: Input image
            
        Returns:
            Image with landmarks drawn
        """
        # Draw face landmarks
        if self.results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                self.results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Draw pose landmarks
        if self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                self.results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw left hand landmarks
        if self.results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                self.results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw right hand landmarks
        if self.results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                self.results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return image
    
    def get_detection_summary(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of what was detected.
        
        Args:
            detection_data: Detection data from detect_holistic()
            
        Returns:
            Summary dictionary
        """
        summary = {
            'hands_detected': 0,
            'pose_detected': detection_data['pose'] is not None,
            'face_detected': detection_data['face'] is not None,
            'hand_details': []
        }
        
        # Count hands and get details
        if detection_data['hands']['left_hand']:
            summary['hands_detected'] += 1
            summary['hand_details'].append('left')
        
        if detection_data['hands']['right_hand']:
            summary['hands_detected'] += 1
            summary['hand_details'].append('right')
        
        # Calculate overall detection quality
        confidence_scores = list(detection_data['detection_confidence'].values())
        summary['overall_confidence'] = sum(confidence_scores) / len(confidence_scores)
        
        return summary
    
    def draw_info(self, image: np.ndarray, detection_data: Dict[str, Any]) -> np.ndarray:
        """
        Draw detection information on the image.
        
        Args:
            image: Input image
            detection_data: Detection data from detect_holistic()
            
        Returns:
            Image with information overlay
        """
        info_image = image.copy()
        summary = self.get_detection_summary(detection_data)
        
        # Background for text
        overlay = info_image.copy()
        cv2.rectangle(overlay, (0, 0), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, info_image, 0.3, 0, info_image)
        
        # Title
        cv2.putText(info_image, "LESSA - Holistic Detection", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Detection status
        hands_text = f"Hands: {summary['hands_detected']} ({', '.join(summary['hand_details'])})"
        cv2.putText(info_image, hands_text, 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if summary['hands_detected'] > 0 else (128, 128, 128), 1)
        
        pose_color = (0, 255, 0) if summary['pose_detected'] else (128, 128, 128)
        cv2.putText(info_image, f"Body Pose: {'✓' if summary['pose_detected'] else '✗'}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
        
        face_color = (0, 255, 0) if summary['face_detected'] else (128, 128, 128)
        cv2.putText(info_image, f"Face: {'✓' if summary['face_detected'] else '✗'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        
        confidence_text = f"Overall: {summary['overall_confidence']:.2f}"
        cv2.putText(info_image, confidence_text, 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return info_image
    
    def is_detection_complete(self, detection_data: Dict[str, Any], require_all: bool = False) -> bool:
        """
        Check if detection is sufficient for sign language recognition.
        
        Args:
            detection_data: Detection data from detect_holistic()
            require_all: If True, requires all components (hands, pose, face)
            
        Returns:
            True if detection is sufficient
        """
        summary = self.get_detection_summary(detection_data)
        
        if require_all:
            return (summary['hands_detected'] > 0 and 
                   summary['pose_detected'] and 
                   summary['face_detected'])
        else:
            # At minimum, need hands OR pose
            return summary['hands_detected'] > 0 or summary['pose_detected']
    
    def close(self):
        """Clean up resources."""
        if self.holistic:
            self.holistic.close()

# Landmark indices for easy reference
POSE_LANDMARKS = {
    'NOSE': 0, 'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
    'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
    'LEFT_EAR': 7, 'RIGHT_EAR': 8, 'MOUTH_LEFT': 9, 'MOUTH_RIGHT': 10,
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20, 'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28, 'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
}

HAND_LANDMARKS = {
    'WRIST': 0,
    'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
    'INDEX_FINGER_MCP': 5, 'INDEX_FINGER_PIP': 6, 'INDEX_FINGER_DIP': 7, 'INDEX_FINGER_TIP': 8,
    'MIDDLE_FINGER_MCP': 9, 'MIDDLE_FINGER_PIP': 10, 'MIDDLE_FINGER_DIP': 11, 'MIDDLE_FINGER_TIP': 12,
    'RING_FINGER_MCP': 13, 'RING_FINGER_PIP': 14, 'RING_FINGER_DIP': 15, 'RING_FINGER_TIP': 16,
    'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
}