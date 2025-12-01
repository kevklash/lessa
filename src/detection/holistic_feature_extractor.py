"""
Holistic feature extraction from hands, body pose, and face landmarks for LESSA.
Supports comprehensive sign language feature extraction including spatial relationships.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..utils.helpers import calculate_distance, calculate_angle, normalize_landmarks

class HolisticFeatureExtractor:
    """Extract comprehensive features from holistic detection data for LESSA."""
    
    def __init__(self):
        """Initialize holistic feature extractor."""
        # Key landmark indices for quick access
        self.hand_landmarks = {
            'WRIST': 0, 'THUMB_TIP': 4, 'INDEX_TIP': 8, 'MIDDLE_TIP': 12, 'RING_TIP': 16, 'PINKY_TIP': 20
        }
        
        self.pose_landmarks = {
            'NOSE': 0, 'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14, 'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24
        }
        
        # Finger tip indices
        self.fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        # Body regions for spatial analysis
        self.body_regions = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Head area
            'chest': [11, 12],  # Shoulders
            'torso': [11, 12, 23, 24],  # Shoulders + Hips
            'arms': [11, 12, 13, 14, 15, 16]  # Shoulder to wrists
        }
    
    def extract_features(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive features from holistic detection data.
        
        Args:
            detection_data: Holistic detection data from HolisticDetector
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'hand_features': self._extract_hand_features(detection_data['hands']),
            'pose_features': self._extract_pose_features(detection_data['pose']),
            'face_features': self._extract_face_features(detection_data['face']),
            'spatial_relationships': self._extract_spatial_relationships(detection_data),
            'temporal_features': {},  # Will be populated by sequence analysis
            'detection_quality': detection_data['detection_confidence'],
            'metadata': {
                'timestamp': detection_data.get('timestamp'),
                'image_dimensions': detection_data['image_dimensions']
            }
        }
        
        return features
    
    def _extract_hand_features(self, hands_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from both hands."""
        hand_features = {
            'left_hand': None,
            'right_hand': None,
            'two_hand_relationships': None
        }
        
        # Process left hand
        if hands_data['left_hand']:
            hand_features['left_hand'] = self._process_single_hand(
                hands_data['left_hand'], 'left'
            )
        
        # Process right hand
        if hands_data['right_hand']:
            hand_features['right_hand'] = self._process_single_hand(
                hands_data['right_hand'], 'right'
            )
        
        # Process two-hand relationships
        if hands_data['left_hand'] and hands_data['right_hand']:
            hand_features['two_hand_relationships'] = self._calculate_two_hand_relationships(
                hands_data['left_hand'], hands_data['right_hand']
            )
        
        return hand_features
    
    def _process_single_hand(self, hand_data: Dict[str, Any], hand_side: str) -> Dict[str, Any]:
        """Process features for a single hand."""
        landmarks = np.array(hand_data['landmarks'])
        
        if len(landmarks) == 0:
            return None
        
        # Extract 3D coordinates (x, y, z)
        landmarks_3d = landmarks[:, :3] if landmarks.shape[1] >= 3 else landmarks
        
        features = {
            'raw_landmarks': landmarks_3d.flatten(),
            'normalized_landmarks': self._normalize_hand_landmarks(landmarks_3d),
            'finger_distances': self._calculate_finger_distances(landmarks_3d),
            'finger_angles': self._calculate_finger_angles(landmarks_3d),
            'hand_shape': self._analyze_hand_shape(landmarks_3d),
            'hand_orientation': self._calculate_hand_orientation(landmarks_3d),
            'finger_states': self._analyze_finger_states(landmarks_3d),
            'hand_side': hand_side,
            'landmark_count': len(landmarks_3d)
        }
        
        return features
    
    def _extract_pose_features(self, pose_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract features from body pose."""
        if not pose_data:
            return None
        
        landmarks = np.array(pose_data['landmarks'])
        
        if len(landmarks) == 0:
            return None
        
        # Extract 3D coordinates and visibility if available
        landmarks_3d = landmarks[:, :3] if landmarks.shape[1] >= 3 else landmarks
        visibility = landmarks[:, 3] if landmarks.shape[1] >= 4 else None
        
        features = {
            'raw_landmarks': landmarks_3d.flatten(),
            'normalized_landmarks': self._normalize_pose_landmarks(landmarks_3d),
            'body_proportions': self._calculate_body_proportions(landmarks_3d),
            'arm_positions': self._analyze_arm_positions(landmarks_3d),
            'shoulder_orientation': self._calculate_shoulder_orientation(landmarks_3d),
            'torso_center': self._calculate_torso_center(landmarks_3d),
            'visibility_scores': visibility.tolist() if visibility is not None else None,
            'landmark_count': len(landmarks_3d)
        }
        
        return features
    
    def _extract_face_features(self, face_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract features from face landmarks."""
        if not face_data:
            return None
        
        landmarks = np.array(face_data['landmarks'])
        
        if len(landmarks) == 0:
            return None
        
        landmarks_3d = landmarks[:, :3] if landmarks.shape[1] >= 3 else landmarks
        
        # For LESSA, we focus on key facial features that affect sign meaning
        features = {
            'face_center': self._calculate_face_center(landmarks_3d),
            'face_orientation': self._calculate_face_orientation(landmarks_3d),
            'mouth_region': self._analyze_mouth_region(landmarks_3d),
            'eye_region': self._analyze_eye_region(landmarks_3d),
            'facial_bounds': self._calculate_facial_bounds(landmarks_3d),
            'landmark_count': len(landmarks_3d)
        }
        
        return features
    
    def _extract_spatial_relationships(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract spatial relationships between hands, body, and face."""
        relationships = {
            'hands_to_body': {},
            'hands_to_face': {},
            'hand_symmetry': {},
            'overall_posture': {}
        }
        
        # Get key reference points
        reference_points = self._get_reference_points(detection_data)
        
        if not reference_points:
            return relationships
        
        # Analyze hands relative to body regions
        relationships['hands_to_body'] = self._analyze_hands_to_body(
            detection_data['hands'], reference_points
        )
        
        # Analyze hands relative to face
        relationships['hands_to_face'] = self._analyze_hands_to_face(
            detection_data['hands'], reference_points
        )
        
        # Analyze hand symmetry and coordination
        if detection_data['hands']['left_hand'] and detection_data['hands']['right_hand']:
            relationships['hand_symmetry'] = self._analyze_hand_symmetry(
                detection_data['hands']['left_hand'],
                detection_data['hands']['right_hand']
            )
        
        # Overall posture analysis
        relationships['overall_posture'] = self._analyze_overall_posture(reference_points)
        
        return relationships
    
    def _get_reference_points(self, detection_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Get key reference points for spatial analysis."""
        ref_points = {}
        
        # Face reference
        if detection_data['face']:
            face_landmarks = np.array(detection_data['face']['landmarks'])[:, :3]
            ref_points['face_center'] = self._calculate_face_center(face_landmarks)
        
        # Body reference points
        if detection_data['pose']:
            pose_landmarks = np.array(detection_data['pose']['landmarks'])[:, :3]
            
            # Key body points
            if len(pose_landmarks) > max(self.pose_landmarks.values()):
                ref_points['left_shoulder'] = pose_landmarks[self.pose_landmarks['LEFT_SHOULDER']]
                ref_points['right_shoulder'] = pose_landmarks[self.pose_landmarks['RIGHT_SHOULDER']]
                ref_points['torso_center'] = self._calculate_torso_center(pose_landmarks)
                ref_points['nose'] = pose_landmarks[self.pose_landmarks['NOSE']]
        
        # Hand reference points
        if detection_data['hands']['left_hand']:
            left_landmarks = np.array(detection_data['hands']['left_hand']['landmarks'])[:, :3]
            ref_points['left_hand_center'] = left_landmarks[0]  # Wrist
        
        if detection_data['hands']['right_hand']:
            right_landmarks = np.array(detection_data['hands']['right_hand']['landmarks'])[:, :3]
            ref_points['right_hand_center'] = right_landmarks[0]  # Wrist
        
        return ref_points
    
    def _normalize_hand_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize hand landmarks relative to wrist and hand size."""
        if len(landmarks) == 0:
            return landmarks
        
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        # Scale by hand size (wrist to middle finger tip distance)
        hand_size = np.linalg.norm(normalized[12])  # Middle finger tip
        if hand_size > 0:
            normalized = normalized / hand_size
        
        return normalized.flatten()
    
    def _normalize_pose_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize pose landmarks relative to torso center and size."""
        if len(landmarks) == 0 or len(landmarks) <= max(self.pose_landmarks.values()):
            return landmarks.flatten()
        
        # Calculate torso center
        torso_center = self._calculate_torso_center(landmarks)
        normalized = landmarks - torso_center
        
        # Scale by torso size (shoulder width)
        left_shoulder = landmarks[self.pose_landmarks['LEFT_SHOULDER']]
        right_shoulder = landmarks[self.pose_landmarks['RIGHT_SHOULDER']]
        torso_size = np.linalg.norm(right_shoulder - left_shoulder)
        
        if torso_size > 0:
            normalized = normalized / torso_size
        
        return normalized.flatten()
    
    def _calculate_torso_center(self, pose_landmarks: np.ndarray) -> np.ndarray:
        """Calculate the center point of the torso."""
        if len(pose_landmarks) <= max(self.pose_landmarks.values()):
            return np.array([0, 0, 0])
        
        # Average of shoulders and hips
        left_shoulder = pose_landmarks[self.pose_landmarks['LEFT_SHOULDER']]
        right_shoulder = pose_landmarks[self.pose_landmarks['RIGHT_SHOULDER']]
        left_hip = pose_landmarks[self.pose_landmarks['LEFT_HIP']]
        right_hip = pose_landmarks[self.pose_landmarks['RIGHT_HIP']]
        
        return (left_shoulder + right_shoulder + left_hip + right_hip) / 4
    
    def _calculate_face_center(self, face_landmarks: np.ndarray) -> np.ndarray:
        """Calculate the center point of the face."""
        if len(face_landmarks) == 0:
            return np.array([0, 0, 0])
        
        # Use mean of all face landmarks as center
        return np.mean(face_landmarks, axis=0)
    
    def _analyze_hands_to_body(self, hands_data: Dict[str, Any], 
                              ref_points: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze hand positions relative to body regions."""
        analysis = {
            'left_hand_to_body': {},
            'right_hand_to_body': {}
        }
        
        if 'torso_center' not in ref_points:
            return analysis
        
        torso_center = ref_points['torso_center']
        
        # Analyze left hand
        if hands_data['left_hand'] and 'left_hand_center' in ref_points:
            left_hand_pos = ref_points['left_hand_center']
            analysis['left_hand_to_body'] = {
                'distance_to_torso': np.linalg.norm(left_hand_pos - torso_center),
                'relative_position': (left_hand_pos - torso_center).tolist(),
                'height_level': self._classify_hand_height(left_hand_pos, ref_points)
            }
        
        # Analyze right hand
        if hands_data['right_hand'] and 'right_hand_center' in ref_points:
            right_hand_pos = ref_points['right_hand_center']
            analysis['right_hand_to_body'] = {
                'distance_to_torso': np.linalg.norm(right_hand_pos - torso_center),
                'relative_position': (right_hand_pos - torso_center).tolist(),
                'height_level': self._classify_hand_height(right_hand_pos, ref_points)
            }
        
        return analysis
    
    def _classify_hand_height(self, hand_pos: np.ndarray, 
                             ref_points: Dict[str, np.ndarray]) -> str:
        """Classify hand height relative to body regions."""
        if 'nose' in ref_points and 'left_shoulder' in ref_points:
            nose_y = ref_points['nose'][1]
            shoulder_y = ref_points['left_shoulder'][1]
            hand_y = hand_pos[1]
            
            if hand_y < nose_y:
                return 'above_head'
            elif hand_y < shoulder_y:
                return 'head_level'
            elif hand_y < shoulder_y + 0.3:  # Approximate chest level
                return 'chest_level'
            else:
                return 'below_chest'
        
        return 'unknown'
    
    def _calculate_finger_distances(self, landmarks: np.ndarray) -> List[float]:
        """Calculate distances between finger landmarks."""
        distances = []
        
        if len(landmarks) < 21:  # Not enough landmarks
            return distances
        
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
        
        return distances
    
    def _calculate_finger_angles(self, landmarks: np.ndarray) -> List[float]:
        """Calculate angles at finger joints."""
        angles = []
        
        if len(landmarks) < 21:
            return angles
        
        # Define finger joint sequences
        finger_joints = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        for finger_name, joints in finger_joints.items():
            for i in range(1, len(joints) - 1):
                point1 = landmarks[joints[i - 1]]
                point2 = landmarks[joints[i]]
                point3 = landmarks[joints[i + 1]]
                
                v1 = point1 - point2
                v2 = point3 - point2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(np.degrees(angle))
        
        return angles
    
    def _analyze_hand_shape(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Analyze overall hand shape characteristics."""
        if len(landmarks) < 21:
            return {}
        
        # Calculate hand openness (spread of fingers)
        fingertip_positions = [landmarks[i] for i in self.fingertips]
        hand_span = max([np.linalg.norm(pos - landmarks[0]) for pos in fingertip_positions])
        
        # Calculate finger curl (how bent fingers are)
        finger_curl_scores = []
        for tip_idx in self.fingertips[1:]:  # Skip thumb
            tip_to_wrist = np.linalg.norm(landmarks[tip_idx] - landmarks[0])
            extended_length = np.linalg.norm(landmarks[tip_idx - 1] - landmarks[0])  # MCP to wrist
            curl_ratio = tip_to_wrist / (extended_length + 1e-8)
            finger_curl_scores.append(curl_ratio)
        
        return {
            'hand_span': hand_span,
            'average_finger_curl': np.mean(finger_curl_scores),
            'finger_curl_variance': np.var(finger_curl_scores),
            'hand_compactness': np.mean(finger_curl_scores) / (hand_span + 1e-8)
        }
    
    def _calculate_hand_orientation(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate hand orientation in 3D space."""
        if len(landmarks) < 21:
            return {}
        
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        orientation_vector = middle_mcp - wrist
        
        # Calculate Euler angles
        pitch = np.arctan2(orientation_vector[2], orientation_vector[1])
        yaw = np.arctan2(orientation_vector[0], orientation_vector[2])
        roll = np.arctan2(orientation_vector[1], orientation_vector[0])
        
        return {
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw),
            'roll': np.degrees(roll)
        }
    
    def _analyze_finger_states(self, landmarks: np.ndarray) -> Dict[str, str]:
        """Analyze the state of each finger (extended, bent, etc.)."""
        if len(landmarks) < 21:
            return {}
        
        finger_states = {}
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, name in enumerate(finger_names):
            tip_idx = self.fingertips[i]
            # Simple heuristic: if tip is far from palm, finger is extended
            tip_to_wrist = np.linalg.norm(landmarks[tip_idx] - landmarks[0])
            
            # Threshold based on typical hand proportions
            if tip_to_wrist > 0.15:  # Adjusted threshold
                finger_states[name] = 'extended'
            else:
                finger_states[name] = 'bent'
        
        return finger_states
    
    def get_comprehensive_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Get a comprehensive feature vector for machine learning."""
        feature_vector = []
        
        # Hand features
        for hand_side in ['left_hand', 'right_hand']:
            hand_data = features['hand_features'].get(hand_side)
            if hand_data:
                feature_vector.extend(hand_data['normalized_landmarks'])
                feature_vector.extend(hand_data['finger_distances'])
                feature_vector.extend(hand_data['finger_angles'])
                
                # Hand shape features
                shape_data = hand_data['hand_shape']
                feature_vector.extend([
                    shape_data.get('hand_span', 0),
                    shape_data.get('average_finger_curl', 0),
                    shape_data.get('hand_compactness', 0)
                ])
                
                # Hand orientation
                orientation = hand_data['hand_orientation']
                feature_vector.extend([
                    orientation.get('pitch', 0),
                    orientation.get('yaw', 0),
                    orientation.get('roll', 0)
                ])
            else:
                # Pad with zeros if hand not detected
                feature_vector.extend([0] * 100)  # Approximate feature count
        
        # Pose features (simplified)
        if features['pose_features']:
            pose_data = features['pose_features']
            feature_vector.extend(pose_data['normalized_landmarks'][:30])  # First 30 elements
        else:
            feature_vector.extend([0] * 30)
        
        # Spatial relationship features
        spatial_data = features['spatial_relationships']
        if 'hands_to_body' in spatial_data:
            for hand_side in ['left_hand_to_body', 'right_hand_to_body']:
                hand_body_data = spatial_data['hands_to_body'].get(hand_side, {})
                feature_vector.extend([
                    hand_body_data.get('distance_to_torso', 0)
                ])
                
                # Add relative position
                rel_pos = hand_body_data.get('relative_position', [0, 0, 0])
                feature_vector.extend(rel_pos)
        
        return np.array(feature_vector)