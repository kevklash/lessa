"""
Temporal feature extraction for dynamic gesture recognition.
Extracts spatial, temporal, and geometric features from gesture sequences.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import interpolate
from scipy.signal import savgol_filter
import math


class TemporalFeatureExtractor:
    """Extract multi-dimensional features from temporal gesture sequences."""
    
    def __init__(self):
        """Initialize temporal feature extractor."""
        # Feature dimensions
        self.spatial_dims = 63  # 21 hand landmarks * 3 coords
        self.velocity_dims = 63  # Same as spatial
        self.acceleration_dims = 63  # Same as spatial
        self.geometric_dims = 15  # Custom geometric features
        
        # Processing parameters
        self.normalize_sequence_length = 30  # Standard sequence length
        self.smoothing_window = 5  # For Savitzky-Golay filtering
        
        # Key landmark indices for hand analysis
        self.fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.finger_bases = [1, 5, 9, 13, 17]  # Finger base joints
        self.wrist_idx = 0
        
    def extract_features(self, gesture_sequence: List[Dict]) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract comprehensive features from gesture sequence.
        
        Args:
            gesture_sequence: List of frame data with detection_data
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            # Extract hand landmarks sequence
            landmarks_sequence = self._extract_landmarks_sequence(gesture_sequence)
            
            if landmarks_sequence is None or len(landmarks_sequence) < 3:
                print("❌ Insufficient landmark data for feature extraction")
                return None
                
            # Normalize sequence length
            normalized_sequence = self._normalize_sequence_length(landmarks_sequence)
            
            # Apply smoothing to reduce noise
            smoothed_sequence = self._apply_temporal_smoothing(normalized_sequence)
            
            # Extract different types of features
            features = {
                'spatial_features': self._extract_spatial_features(smoothed_sequence),
                'temporal_features': self._extract_temporal_features(smoothed_sequence),
                'geometric_features': self._extract_geometric_features(smoothed_sequence),
                'motion_features': self._extract_motion_features(smoothed_sequence),
                'trajectory_features': self._extract_trajectory_features(smoothed_sequence)
            }
            
            # Flatten and concatenate all features
            feature_vector = self._concatenate_features(features)
            
            return {
                'feature_vector': feature_vector,
                'individual_features': features,
                'sequence_info': {
                    'original_length': len(landmarks_sequence),
                    'normalized_length': len(smoothed_sequence),
                    'hand_type': self._detect_hand_type(smoothed_sequence)
                }
            }
            
        except Exception as e:
            print(f"❌ Error extracting temporal features: {e}")
            return None
            
    def _extract_landmarks_sequence(self, gesture_sequence: List[Dict]) -> Optional[np.ndarray]:
        """Extract hand landmarks from gesture sequence."""
        try:
            landmarks_list = []
            
            for frame_data in gesture_sequence:
                detection_data = frame_data.get('detection_data', {})
                hands_data = detection_data.get('hands', {})
                
                # Get active hand (prioritize right hand)
                hand_data = (hands_data.get('right_hand') or 
                           hands_data.get('left_hand'))
                
                if hand_data and hand_data.get('landmarks'):
                    landmarks = np.array(hand_data['landmarks'])[:, :3]  # x, y, z
                    if len(landmarks) >= 21:
                        landmarks_list.append(landmarks)
                    else:
                        # Pad with zeros if insufficient landmarks
                        padded = np.zeros((21, 3))
                        padded[:len(landmarks)] = landmarks
                        landmarks_list.append(padded)
                else:
                    # No hand detected, use zeros
                    landmarks_list.append(np.zeros((21, 3)))
                    
            if len(landmarks_list) == 0:
                return None
                
            return np.array(landmarks_list)  # Shape: (sequence_length, 21, 3)
            
        except Exception as e:
            print(f"❌ Error extracting landmarks sequence: {e}")
            return None
            
    def _normalize_sequence_length(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """Normalize sequence to standard length using interpolation."""
        try:
            original_length = len(landmarks_sequence)
            
            if original_length == self.normalize_sequence_length:
                return landmarks_sequence
                
            # Create interpolation functions for each landmark coordinate
            normalized_sequence = np.zeros((self.normalize_sequence_length, 21, 3))
            
            for landmark_idx in range(21):
                for coord_idx in range(3):
                    # Original time points
                    original_time = np.linspace(0, 1, original_length)
                    # New time points
                    new_time = np.linspace(0, 1, self.normalize_sequence_length)
                    
                    # Interpolate
                    values = landmarks_sequence[:, landmark_idx, coord_idx]
                    interp_func = interpolate.interp1d(original_time, values, 
                                                     kind='linear', 
                                                     fill_value='extrapolate')
                    normalized_sequence[:, landmark_idx, coord_idx] = interp_func(new_time)
                    
            return normalized_sequence
            
        except Exception as e:
            print(f"❌ Error normalizing sequence length: {e}")
            return landmarks_sequence
            
    def _apply_temporal_smoothing(self, sequence: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce noise."""
        try:
            if len(sequence) < self.smoothing_window:
                return sequence
                
            smoothed = np.copy(sequence)
            
            for landmark_idx in range(21):
                for coord_idx in range(3):
                    # Apply Savitzky-Golay filter
                    values = sequence[:, landmark_idx, coord_idx]
                    if np.std(values) > 1e-6:  # Only smooth if there's variation
                        smoothed_values = savgol_filter(values, 
                                                      min(self.smoothing_window, len(values)), 
                                                      2)  # 2nd order polynomial
                        smoothed[:, landmark_idx, coord_idx] = smoothed_values
                        
            return smoothed
            
        except Exception as e:
            print(f"❌ Error applying smoothing: {e}")
            return sequence
            
    def _extract_spatial_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract spatial features (normalized positions)."""
        try:
            # Normalize relative to wrist for each frame
            spatial_features = []
            
            for frame_landmarks in sequence:
                wrist = frame_landmarks[self.wrist_idx]
                normalized = frame_landmarks - wrist
                
                # Scale by hand size (middle finger length)
                middle_finger_length = np.linalg.norm(normalized[12]) + 1e-8
                normalized = normalized / middle_finger_length
                
                spatial_features.append(normalized.flatten())
                
            return np.array(spatial_features)  # Shape: (seq_len, 63)
            
        except Exception as e:
            print(f"❌ Error extracting spatial features: {e}")
            return np.zeros((len(sequence), self.spatial_dims))
            
    def _extract_temporal_features(self, sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract temporal features (velocity, acceleration)."""
        try:
            # Calculate velocities (first derivative)
            velocities = np.zeros_like(sequence)
            velocities[1:] = sequence[1:] - sequence[:-1]
            
            # Calculate accelerations (second derivative)
            accelerations = np.zeros_like(sequence)
            accelerations[1:] = velocities[1:] - velocities[:-1]
            
            # Flatten for each frame
            velocity_features = []
            acceleration_features = []
            
            for i in range(len(sequence)):
                velocity_features.append(velocities[i].flatten())
                acceleration_features.append(accelerations[i].flatten())
                
            return {
                'velocities': np.array(velocity_features),
                'accelerations': np.array(acceleration_features)
            }
            
        except Exception as e:
            print(f"❌ Error extracting temporal features: {e}")
            return {
                'velocities': np.zeros((len(sequence), self.velocity_dims)),
                'accelerations': np.zeros((len(sequence), self.acceleration_dims))
            }
            
    def _extract_geometric_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract geometric features (angles, distances, ratios)."""
        try:
            geometric_features = []
            
            for frame_landmarks in sequence:
                frame_features = []
                
                # Normalize relative to wrist
                wrist = frame_landmarks[self.wrist_idx]
                normalized = frame_landmarks - wrist
                
                # 1. Finger curl ratios (5 features)
                for tip_idx in self.fingertips:
                    tip_distance = np.linalg.norm(normalized[tip_idx])
                    frame_features.append(tip_distance)
                    
                # 2. Finger spreads (4 features)
                for i in range(len(self.fingertips) - 1):
                    spread = np.linalg.norm(normalized[self.fingertips[i]] - 
                                         normalized[self.fingertips[i+1]])
                    frame_features.append(spread)
                    
                # 3. Hand openness indicators (3 features)
                thumb_index_dist = np.linalg.norm(normalized[4] - normalized[8])
                palm_center = np.mean([normalized[5], normalized[9], 
                                     normalized[13], normalized[17]], axis=0)
                avg_tip_distance = np.mean([np.linalg.norm(normalized[tip] - palm_center) 
                                          for tip in self.fingertips])
                hand_span = np.linalg.norm(normalized[4] - normalized[20])
                
                frame_features.extend([thumb_index_dist, avg_tip_distance, hand_span])
                
                # 4. Finger angles (3 features - key angles)
                # Thumb angle
                thumb_angle = self._calculate_angle(normalized[2], normalized[3], normalized[4])
                # Index finger angle
                index_angle = self._calculate_angle(normalized[6], normalized[7], normalized[8])
                # Middle finger angle
                middle_angle = self._calculate_angle(normalized[10], normalized[11], normalized[12])
                
                frame_features.extend([thumb_angle, index_angle, middle_angle])
                
                geometric_features.append(frame_features)
                
            return np.array(geometric_features)  # Shape: (seq_len, 15)
            
        except Exception as e:
            print(f"❌ Error extracting geometric features: {e}")
            return np.zeros((len(sequence), self.geometric_dims))
            
    def _extract_motion_features(self, sequence: np.ndarray) -> Dict[str, float]:
        """Extract global motion characteristics."""
        try:
            # Calculate overall motion magnitude
            total_motion = 0.0
            motion_changes = []
            
            for i in range(1, len(sequence)):
                frame_motion = np.mean(np.linalg.norm(sequence[i] - sequence[i-1], axis=1))
                total_motion += frame_motion
                motion_changes.append(frame_motion)
                
            avg_motion = total_motion / max(1, len(sequence) - 1)
            motion_variance = np.var(motion_changes) if motion_changes else 0.0
            
            # Calculate dominant direction
            start_point = np.mean(sequence[0], axis=0)  # Average of all landmarks
            end_point = np.mean(sequence[-1], axis=0)
            direction_vector = end_point - start_point
            
            return {
                'avg_motion': avg_motion,
                'motion_variance': motion_variance,
                'total_displacement': np.linalg.norm(direction_vector),
                'direction_x': direction_vector[0],
                'direction_y': direction_vector[1],
                'direction_z': direction_vector[2]
            }
            
        except Exception as e:
            print(f"❌ Error extracting motion features: {e}")
            return {
                'avg_motion': 0.0, 'motion_variance': 0.0, 
                'total_displacement': 0.0, 'direction_x': 0.0,
                'direction_y': 0.0, 'direction_z': 0.0
            }
            
    def _extract_trajectory_features(self, sequence: np.ndarray) -> Dict[str, Any]:
        """Extract trajectory-specific features for letters like J and Z."""
        try:
            # Focus on index fingertip trajectory (most important for letter tracing)
            index_tip_trajectory = sequence[:, 8, :]  # Index fingertip
            
            # Calculate trajectory characteristics
            trajectory_length = 0.0
            direction_changes = 0
            curvature_points = []
            
            for i in range(1, len(index_tip_trajectory)):
                # Segment length
                segment_length = np.linalg.norm(index_tip_trajectory[i] - 
                                              index_tip_trajectory[i-1])
                trajectory_length += segment_length
                
                # Direction changes (for detecting corners in Z or curves in J)
                if i > 1:
                    v1 = index_tip_trajectory[i-1] - index_tip_trajectory[i-2]
                    v2 = index_tip_trajectory[i] - index_tip_trajectory[i-1]
                    
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = math.acos(cos_angle)
                        
                        # Significant direction change
                        if angle > math.pi / 4:  # 45 degrees
                            direction_changes += 1
                            curvature_points.append(i)
                            
            return {
                'trajectory_length': trajectory_length,
                'direction_changes': direction_changes,
                'avg_segment_length': trajectory_length / max(1, len(index_tip_trajectory) - 1),
                'curvature_points': len(curvature_points),
                'start_position': index_tip_trajectory[0].tolist(),
                'end_position': index_tip_trajectory[-1].tolist()
            }
            
        except Exception as e:
            print(f"❌ Error extracting trajectory features: {e}")
            return {
                'trajectory_length': 0.0, 'direction_changes': 0,
                'avg_segment_length': 0.0, 'curvature_points': 0,
                'start_position': [0, 0, 0], 'end_position': [0, 0, 0]
            }
            
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            return math.acos(cos_angle)
            
        except Exception:
            return 0.0
            
    def _detect_hand_type(self, sequence: np.ndarray) -> str:
        """Detect if left or right hand based on thumb position."""
        try:
            # Average thumb and index finger positions
            thumb_pos = np.mean(sequence[:, 4, :], axis=0)  # Thumb tip
            index_pos = np.mean(sequence[:, 8, :], axis=0)  # Index tip
            
            # If thumb is to the right of index, likely right hand
            if thumb_pos[0] > index_pos[0]:
                return 'right'
            else:
                return 'left'
                
        except Exception:
            return 'unknown'
            
    def _concatenate_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Concatenate all features into a single vector."""
        try:
            feature_parts = []
            
            # Spatial features (flattened across time)
            spatial = features['spatial_features'].flatten()
            feature_parts.append(spatial)
            
            # Temporal features (flattened across time)
            velocities = features['temporal_features']['velocities'].flatten()
            accelerations = features['temporal_features']['accelerations'].flatten()
            feature_parts.extend([velocities, accelerations])
            
            # Geometric features (flattened across time)
            geometric = features['geometric_features'].flatten()
            feature_parts.append(geometric)
            
            # Motion features (single values)
            motion = features['motion_features']
            motion_vector = np.array([
                motion['avg_motion'], motion['motion_variance'],
                motion['total_displacement'], motion['direction_x'],
                motion['direction_y'], motion['direction_z']
            ])
            feature_parts.append(motion_vector)
            
            # Trajectory features (single values)
            trajectory = features['trajectory_features']
            trajectory_vector = np.array([
                trajectory['trajectory_length'], trajectory['direction_changes'],
                trajectory['avg_segment_length'], trajectory['curvature_points']
            ] + trajectory['start_position'] + trajectory['end_position'])
            feature_parts.append(trajectory_vector)
            
            # Concatenate all parts
            return np.concatenate(feature_parts)
            
        except Exception as e:
            print(f"❌ Error concatenating features: {e}")
            return np.array([])
            
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get the dimensions of different feature types."""
        return {
            'spatial': self.spatial_dims * self.normalize_sequence_length,
            'velocity': self.velocity_dims * self.normalize_sequence_length,
            'acceleration': self.acceleration_dims * self.normalize_sequence_length,
            'geometric': self.geometric_dims * self.normalize_sequence_length,
            'motion': 6,
            'trajectory': 10,
            'total': (self.spatial_dims + self.velocity_dims + self.acceleration_dims + 
                     self.geometric_dims) * self.normalize_sequence_length + 16
        }