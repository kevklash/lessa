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
    
    def _calculate_body_proportions(self, pose_landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate body proportions for pose analysis."""
        if len(pose_landmarks) == 0 or len(pose_landmarks) <= max(self.pose_landmarks.values()):
            return {}
        
        proportions = {}
        
        try:
            # Get key landmarks
            left_shoulder = pose_landmarks[self.pose_landmarks['LEFT_SHOULDER']]
            right_shoulder = pose_landmarks[self.pose_landmarks['RIGHT_SHOULDER']]
            left_hip = pose_landmarks[self.pose_landmarks['LEFT_HIP']]
            right_hip = pose_landmarks[self.pose_landmarks['RIGHT_HIP']]
            left_elbow = pose_landmarks[self.pose_landmarks['LEFT_ELBOW']]
            right_elbow = pose_landmarks[self.pose_landmarks['RIGHT_ELBOW']]
            left_wrist = pose_landmarks[self.pose_landmarks['LEFT_WRIST']]
            right_wrist = pose_landmarks[self.pose_landmarks['RIGHT_WRIST']]
            nose = pose_landmarks[self.pose_landmarks['NOSE']]
            
            # Calculate basic distances
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            hip_width = np.linalg.norm(right_hip - left_hip)
            torso_height = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
            
            # Arm lengths
            left_upper_arm = np.linalg.norm(left_shoulder - left_elbow)
            right_upper_arm = np.linalg.norm(right_shoulder - right_elbow)
            left_forearm = np.linalg.norm(left_elbow - left_wrist)
            right_forearm = np.linalg.norm(right_elbow - right_wrist)
            
            # Head to body distance
            head_to_shoulders = np.linalg.norm(nose - (left_shoulder + right_shoulder) / 2)
            
            # Store proportions
            proportions.update({
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'torso_height': torso_height,
                'shoulder_hip_ratio': shoulder_width / (hip_width + 1e-8),
                'left_upper_arm_length': left_upper_arm,
                'right_upper_arm_length': right_upper_arm,
                'left_forearm_length': left_forearm,
                'right_forearm_length': right_forearm,
                'arm_symmetry': abs(left_upper_arm - right_upper_arm) / (max(left_upper_arm, right_upper_arm) + 1e-8),
                'head_to_shoulders_distance': head_to_shoulders,
                'torso_aspect_ratio': torso_height / (shoulder_width + 1e-8)
            })
            
        except (IndexError, KeyError) as e:
            # If we can't calculate some proportions, return what we can
            print(f"Warning: Could not calculate some body proportions: {e}")
        
        return proportions
    
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
    
    # Additional missing methods
    
    def _calculate_two_hand_relationships(self, left_hand: Dict[str, Any], 
                                        right_hand: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relationships between two hands."""
        left_landmarks = np.array(left_hand['landmarks'])[:, :3]
        right_landmarks = np.array(right_hand['landmarks'])[:, :3]
        
        # Distance between wrists
        wrist_distance = np.linalg.norm(left_landmarks[0] - right_landmarks[0])
        
        # Hand symmetry analysis
        symmetry_score = self._calculate_hand_symmetry_score(left_landmarks, right_landmarks)
        
        return {
            'wrist_distance': wrist_distance,
            'symmetry_score': symmetry_score,
            'relative_orientation': self._calculate_relative_hand_orientation(left_landmarks, right_landmarks)
        }
    
    def _calculate_hand_symmetry_score(self, left_landmarks: np.ndarray, 
                                     right_landmarks: np.ndarray) -> float:
        """Calculate symmetry score between two hands."""
        if len(left_landmarks) < 21 or len(right_landmarks) < 21:
            return 0.0
        
        # Mirror right hand and compare with left hand
        mirrored_right = right_landmarks.copy()
        mirrored_right[:, 0] *= -1  # Mirror x-coordinate
        
        # Calculate average distance between corresponding landmarks
        distances = [np.linalg.norm(left_landmarks[i] - mirrored_right[i]) for i in range(21)]
        return 1.0 / (1.0 + np.mean(distances))
    
    def _calculate_relative_hand_orientation(self, left_landmarks: np.ndarray,
                                           right_landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate relative orientation between hands."""
        left_orientation = self._calculate_hand_orientation(left_landmarks)
        right_orientation = self._calculate_hand_orientation(right_landmarks)
        
        return {
            'pitch_difference': abs(left_orientation.get('pitch', 0) - right_orientation.get('pitch', 0)),
            'yaw_difference': abs(left_orientation.get('yaw', 0) - right_orientation.get('yaw', 0)),
            'roll_difference': abs(left_orientation.get('roll', 0) - right_orientation.get('roll', 0))
        }
    
    def _analyze_arm_positions(self, pose_landmarks: np.ndarray) -> Dict[str, Any]:
        """Analyze arm positions and angles."""
        if len(pose_landmarks) <= max(self.pose_landmarks.values()):
            return {}
        
        try:
            left_shoulder = pose_landmarks[self.pose_landmarks['LEFT_SHOULDER']]
            right_shoulder = pose_landmarks[self.pose_landmarks['RIGHT_SHOULDER']]
            left_elbow = pose_landmarks[self.pose_landmarks['LEFT_ELBOW']]
            right_elbow = pose_landmarks[self.pose_landmarks['RIGHT_ELBOW']]
            left_wrist = pose_landmarks[self.pose_landmarks['LEFT_WRIST']]
            right_wrist = pose_landmarks[self.pose_landmarks['RIGHT_WRIST']]
            
            # Calculate arm angles
            left_arm_angle = self._calculate_joint_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self._calculate_joint_angle(right_shoulder, right_elbow, right_wrist)
            
            return {
                'left_arm_angle': left_arm_angle,
                'right_arm_angle': right_arm_angle,
                'arm_angle_difference': abs(left_arm_angle - right_arm_angle),
                'left_elbow_height': left_elbow[1] - left_shoulder[1],
                'right_elbow_height': right_elbow[1] - right_shoulder[1]
            }
        except (IndexError, KeyError):
            return {}
    
    def _calculate_joint_angle(self, point1: np.ndarray, joint: np.ndarray, 
                             point3: np.ndarray) -> float:
        """Calculate angle at a joint between three points."""
        v1 = point1 - joint
        v2 = point3 - joint
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _calculate_shoulder_orientation(self, pose_landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate shoulder orientation."""
        if len(pose_landmarks) <= max(self.pose_landmarks.values()):
            return {}
        
        try:
            left_shoulder = pose_landmarks[self.pose_landmarks['LEFT_SHOULDER']]
            right_shoulder = pose_landmarks[self.pose_landmarks['RIGHT_SHOULDER']]
            
            shoulder_vector = right_shoulder - left_shoulder
            
            # Calculate shoulder tilt angle
            tilt_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
            
            return {
                'shoulder_tilt_degrees': np.degrees(tilt_angle),
                'shoulder_width': np.linalg.norm(shoulder_vector)
            }
        except (IndexError, KeyError):
            return {}
    
    def _calculate_face_orientation(self, face_landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate face orientation from landmarks."""
        if len(face_landmarks) < 10:
            return {}
        
        # Use a subset of face landmarks to estimate orientation
        # This is a simplified approach - in practice, you might want more sophisticated methods
        face_center = self._calculate_face_center(face_landmarks)
        
        # Estimate nose tip (approximate)
        nose_area = face_landmarks[:10]  # First few landmarks are usually around nose
        nose_tip = np.mean(nose_area, axis=0)
        
        orientation_vector = nose_tip - face_center
        
        pitch = np.arctan2(orientation_vector[1], orientation_vector[2])
        yaw = np.arctan2(orientation_vector[0], orientation_vector[2])
        
        return {
            'face_pitch': np.degrees(pitch),
            'face_yaw': np.degrees(yaw)
        }
    
    def _analyze_mouth_region(self, face_landmarks: np.ndarray) -> Dict[str, Any]:
        """Analyze mouth region features."""
        if len(face_landmarks) < 20:
            return {}
        
        # MediaPipe face landmarks for mouth are typically in a specific range
        # This is a simplified approach - actual indices would depend on MediaPipe spec
        mouth_landmarks = face_landmarks[10:20]  # Approximate mouth region
        
        mouth_center = np.mean(mouth_landmarks, axis=0)
        mouth_width = np.max(mouth_landmarks[:, 0]) - np.min(mouth_landmarks[:, 0])
        mouth_height = np.max(mouth_landmarks[:, 1]) - np.min(mouth_landmarks[:, 1])
        
        return {
            'mouth_center': mouth_center.tolist(),
            'mouth_width': mouth_width,
            'mouth_height': mouth_height,
            'mouth_aspect_ratio': mouth_width / (mouth_height + 1e-8)
        }
    
    def _analyze_eye_region(self, face_landmarks: np.ndarray) -> Dict[str, Any]:
        """Analyze eye region features."""
        if len(face_landmarks) < 10:
            return {}
        
        # Simplified eye region analysis
        eye_landmarks = face_landmarks[0:10]  # Approximate eye region
        
        eye_center = np.mean(eye_landmarks, axis=0)
        eye_span = np.max(eye_landmarks[:, 0]) - np.min(eye_landmarks[:, 0])
        
        return {
            'eye_center': eye_center.tolist(),
            'eye_span': eye_span
        }
    
    def _calculate_facial_bounds(self, face_landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate bounding box of face landmarks."""
        if len(face_landmarks) == 0:
            return {}
        
        min_coords = np.min(face_landmarks, axis=0)
        max_coords = np.max(face_landmarks, axis=0)
        
        return {
            'face_width': max_coords[0] - min_coords[0],
            'face_height': max_coords[1] - min_coords[1],
            'face_depth': max_coords[2] - min_coords[2] if len(max_coords) > 2 else 0
        }
    
    def _analyze_hands_to_face(self, hands_data: Dict[str, Any],
                             ref_points: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze hand positions relative to face."""
        analysis = {}
        
        if 'face_center' not in ref_points:
            return analysis
        
        face_center = ref_points['face_center']
        
        # Analyze each hand relative to face
        for hand_side, hand_key in [('left', 'left_hand_center'), ('right', 'right_hand_center')]:
            if hand_key in ref_points:
                hand_pos = ref_points[hand_key]
                distance_to_face = np.linalg.norm(hand_pos - face_center)
                
                analysis[f'{hand_side}_hand_to_face'] = {
                    'distance': distance_to_face,
                    'relative_position': (hand_pos - face_center).tolist()
                }
        
        return analysis
    
    def _analyze_hand_symmetry(self, left_hand: Dict[str, Any], 
                             right_hand: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze symmetry between hands."""
        left_landmarks = np.array(left_hand['landmarks'])[:, :3]
        right_landmarks = np.array(right_hand['landmarks'])[:, :3]
        
        symmetry_score = self._calculate_hand_symmetry_score(left_landmarks, right_landmarks)
        
        return {
            'symmetry_score': symmetry_score,
            'is_symmetric': symmetry_score > 0.7  # Threshold for symmetry
        }
    
    def _analyze_overall_posture(self, ref_points: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze overall body posture."""
        posture_analysis = {}
        
        # Check if we have enough reference points
        required_points = ['torso_center', 'nose']
        if not all(point in ref_points for point in required_points):
            return posture_analysis
        
        torso_center = ref_points['torso_center']
        nose = ref_points['nose']
        
        # Calculate body alignment
        body_vector = nose - torso_center
        posture_analysis['body_alignment'] = {
            'forward_lean': body_vector[2],  # Z-axis lean
            'side_lean': abs(body_vector[0])  # X-axis lean
        }
        
        # Add hand positions if available
        hand_positions = []
        for hand_key in ['left_hand_center', 'right_hand_center']:
            if hand_key in ref_points:
                hand_positions.append(ref_points[hand_key])
        
        if len(hand_positions) >= 2:
            posture_analysis['hand_coordination'] = {
                'hands_distance': np.linalg.norm(hand_positions[0] - hand_positions[1]),
                'hands_height_difference': abs(hand_positions[0][1] - hand_positions[1][1])
            }
        
        return posture_analysis