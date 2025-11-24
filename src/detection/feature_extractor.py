"""
Feature extraction from hand landmarks for sign language recognition.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from ..utils.helpers import calculate_distance, calculate_angle, normalize_landmarks

class FeatureExtractor:
    """Extract features from hand landmarks for machine learning."""
    
    def __init__(self):
        """Initialize feature extractor."""
        # Hand landmark indices (21 landmarks per hand)
        self.landmark_indices = {
            'WRIST': 0,
            'THUMB_TIP': 4,
            'INDEX_TIP': 8,
            'MIDDLE_TIP': 12,
            'RING_TIP': 16,
            'PINKY_TIP': 20
        }
        
        # Finger tip indices for easy access
        self.fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        # Finger joints for angle calculations
        self.finger_joints = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
    
    def extract_features(self, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive features from hand landmarks.
        
        Args:
            hand_data: Dictionary containing hand landmarks and metadata
            
        Returns:
            Dictionary containing extracted features
        """
        landmarks = np.array(hand_data['landmarks'])
        
        features = {
            'raw_landmarks': landmarks.flatten(),
            'normalized_landmarks': self._get_normalized_landmarks(landmarks),
            'distances': self._calculate_distances(landmarks),
            'angles': self._calculate_angles(landmarks),
            'finger_positions': self._get_finger_positions(landmarks),
            'hand_orientation': self._get_hand_orientation(landmarks),
            'gesture_metadata': {
                'hand_label': hand_data.get('label', 'Unknown'),
                'confidence': hand_data.get('confidence', 0.0),
                'num_landmarks': len(landmarks)
            }
        }
        
        return features
    
    def _get_normalized_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks relative to wrist position and hand size."""
        if len(landmarks) == 0:
            return landmarks
        
        # Use wrist as reference point
        wrist = landmarks[0]
        
        # Translate to make wrist the origin
        normalized = landmarks - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        hand_size = np.linalg.norm(normalized[12])  # Middle finger tip
        if hand_size > 0:
            normalized = normalized / hand_size
        
        return normalized.flatten()
    
    def _calculate_distances(self, landmarks: np.ndarray) -> List[float]:
        """Calculate important distances between landmarks."""
        distances = []
        
        # Distance from wrist to each fingertip
        wrist = landmarks[0]
        for tip_idx in self.fingertips:
            distance = np.linalg.norm(landmarks[tip_idx] - wrist)
            distances.append(distance)
        
        # Distances between consecutive fingertips
        for i in range(len(self.fingertips) - 1):
            tip1 = landmarks[self.fingertips[i]]
            tip2 = landmarks[self.fingertips[i + 1]]
            distance = np.linalg.norm(tip2 - tip1)
            distances.append(distance)
        
        # Distance between thumb tip and index tip (important for many signs)
        thumb_index_distance = np.linalg.norm(landmarks[4] - landmarks[8])
        distances.append(thumb_index_distance)
        
        return distances
    
    def _calculate_angles(self, landmarks: np.ndarray) -> List[float]:
        """Calculate angles at finger joints."""
        angles = []
        
        # Calculate angles for each finger
        for finger_name, joints in self.finger_joints.items():
            if len(joints) >= 3:
                # Calculate angle at each joint
                for i in range(1, len(joints) - 1):
                    point1 = landmarks[joints[i - 1]]
                    point2 = landmarks[joints[i]]  # Vertex
                    point3 = landmarks[joints[i + 1]]
                    
                    # Calculate angle using vectors
                    v1 = point1 - point2
                    v2 = point3 - point2
                    
                    # Calculate angle
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    angles.append(np.degrees(angle))
        
        return angles
    
    def _get_finger_positions(self, landmarks: np.ndarray) -> Dict[str, List[float]]:
        """Get relative positions of fingertips."""
        wrist = landmarks[0]
        finger_positions = {}
        
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, finger_name in enumerate(finger_names):
            tip_position = landmarks[self.fingertips[i]] - wrist
            finger_positions[finger_name] = tip_position.tolist()
        
        return finger_positions
    
    def _get_hand_orientation(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate hand orientation features."""
        # Vector from wrist to middle finger MCP
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Calculate hand orientation vector
        orientation_vector = middle_mcp - wrist
        
        # Calculate pitch (rotation around x-axis)
        pitch = np.arctan2(orientation_vector[2], orientation_vector[1])
        
        # Calculate yaw (rotation around y-axis)  
        yaw = np.arctan2(orientation_vector[0], orientation_vector[2])
        
        # Calculate roll (rotation around z-axis)
        roll = np.arctan2(orientation_vector[1], orientation_vector[0])
        
        return {
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw), 
            'roll': np.degrees(roll)
        }
    
    def extract_sequence_features(self, hand_sequence: List[Dict[str, Any]], 
                                sequence_length: int = 30) -> Optional[Dict[str, Any]]:
        """
        Extract features from a sequence of hand landmarks for dynamic gestures.
        
        Args:
            hand_sequence: List of hand data dictionaries over time
            sequence_length: Target length for the sequence
            
        Returns:
            Dictionary containing sequence features
        """
        if not hand_sequence:
            return None
        
        # Extract features for each frame
        frame_features = []
        for hand_data in hand_sequence:
            features = self.extract_features(hand_data)
            # Use normalized landmarks and distances for sequence
            combined_features = np.concatenate([
                features['normalized_landmarks'],
                features['distances']
            ])
            frame_features.append(combined_features)
        
        # Pad or truncate sequence to target length
        frame_features = self._pad_or_truncate_sequence(frame_features, sequence_length)
        
        # Calculate temporal features
        temporal_features = self._calculate_temporal_features(frame_features)
        
        return {
            'sequence_features': np.array(frame_features),
            'temporal_features': temporal_features,
            'sequence_length': len(frame_features)
        }
    
    def _pad_or_truncate_sequence(self, sequence: List[np.ndarray], 
                                target_length: int) -> List[np.ndarray]:
        """Pad or truncate sequence to target length."""
        if len(sequence) == target_length:
            return sequence
        elif len(sequence) > target_length:
            # Truncate by taking evenly spaced frames
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return [sequence[i] for i in indices]
        else:
            # Pad by repeating last frame
            padded_sequence = sequence.copy()
            last_frame = sequence[-1]
            while len(padded_sequence) < target_length:
                padded_sequence.append(last_frame)
            return padded_sequence
    
    def _calculate_temporal_features(self, sequence: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate temporal features from sequence."""
        if len(sequence) < 2:
            return {'velocity': [], 'acceleration': []}
        
        sequence_array = np.array(sequence)
        
        # Calculate velocity (first derivative)
        velocity = np.diff(sequence_array, axis=0)
        
        # Calculate acceleration (second derivative)
        acceleration = np.diff(velocity, axis=0) if len(velocity) > 1 else []
        
        return {
            'velocity': velocity.tolist() if len(velocity) > 0 else [],
            'acceleration': acceleration.tolist() if len(acceleration) > 0 else [],
            'mean_velocity': np.mean(np.abs(velocity)) if len(velocity) > 0 else 0,
            'max_velocity': np.max(np.abs(velocity)) if len(velocity) > 0 else 0
        }
    
    def get_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Get a flat feature vector for machine learning."""
        feature_vector = []
        
        # Add normalized landmarks
        feature_vector.extend(features['normalized_landmarks'])
        
        # Add distances
        feature_vector.extend(features['distances'])
        
        # Add angles
        feature_vector.extend(features['angles'])
        
        # Add hand orientation
        orientation = features['hand_orientation']
        feature_vector.extend([orientation['pitch'], orientation['yaw'], orientation['roll']])
        
        return np.array(feature_vector)