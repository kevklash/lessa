"""
LESSA Alphabet Recognizer - Real-time letter recognition using collected data
Tests pattern recognition against collected alphabet samples.
"""

import cv2
import numpy as np
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detection.holistic_detector import HolisticDetector
from src.detection.holistic_feature_extractor import HolisticFeatureExtractor
from src.utils.enhanced_camera import CameraManager
from src.data.feature_cache import FeatureCache
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class AlphabetRecognizer:
    """Real-time alphabet recognition using collected LESSA data."""
    
    def __init__(self, data_file: str = "lessa_alphabet_data.json"):
        """Initialize the alphabet recognizer."""
        self.data_file = data_file
        self.detector = HolisticDetector()
        self.feature_extractor = HolisticFeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = {}
        self.trained_letters = []
        
        # Feature caching for performance
        self.feature_cache = FeatureCache()
        
        # Recognition parameters
        self.confidence_threshold = 0.3  # Lower threshold for better detection
        self.recognition_history = []
        self.history_size = 5
        
        # Display settings
        self.show_landmarks = True
        self.show_prediction_info = True
        
        # Load and prepare training data
        self.load_training_data()
    
    def load_training_data(self) -> bool:
        """Load and prepare training data using feature cache for performance."""
        load_start_time = time.time()
        
        try:
            if not os.path.exists(self.data_file):
                print(f"⚠️  No training data found at {self.data_file}")
                print("   Please collect some alphabet samples first using the alphabet collector.")
                return False
            
            # Check if we can use cached features
            if self.feature_cache.is_cache_valid(self.data_file):
                print("🚀 Using cached features for fast loading...")
                X, y, features_by_letter = self.feature_cache.load_cache()
                
                if X is not None and y is not None and len(X) > 0:
                    # Update trained letters
                    self.trained_letters = sorted(features_by_letter.keys())
                    
                    # Store training features for similarity calculation
                    self.training_features = X
                    
                    load_time = time.time() - load_start_time
                    print(f"✅ Cached data loaded in {load_time:.3f}s (vs ~{load_time*10:.1f}s without cache)")
                    print(f"   • Letters with data: {', '.join(self.trained_letters)}")
                    print(f"   • Total samples: {len(X)}")
                    print(f"   • Feature dimensions: {X.shape[1]}")
                    
                    # Train the model
                    return self._train_model(X, y)
            
            # Cache is invalid or doesn't exist, rebuild it
            print("🔄 Building feature cache (one-time process)...")
            if not self.feature_cache.build_cache(self.data_file, self):
                return False
            
            # Load the newly built cache
            X, y, features_by_letter = self.feature_cache.load_cache()
            
            if X is None or y is None or len(X) == 0:
                print("❌ No valid training features found.")
                return False
            
            # Update trained letters
            self.trained_letters = sorted(features_by_letter.keys())
            
            # Store training features for similarity calculation
            self.training_features = X
            
            load_time = time.time() - load_start_time
            print(f"📊 Training data processed in {load_time:.2f}s:")
            print(f"   • Letters with data: {', '.join(self.trained_letters)}")
            print(f"   • Total samples: {len(X)}")
            print(f"   • Feature dimensions: {X.shape[1]}")
            
            # Train the model
            return self._train_model(X, y)
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
    
    def _extract_features_from_sample(self, sample: Dict) -> Optional[np.ndarray]:
        """Extract hand-focused feature vector from a training sample."""
        try:
            detection_data = sample.get('detection_data', {})
            hands_data = detection_data.get('hands', {})
            
            # Focus on hand features only (ignore pose and face for alphabet recognition)
            feature_vector = []
            
            # Check if we have any hand data
            left_hand_data = hands_data.get('left_hand')
            right_hand_data = hands_data.get('right_hand')
            
            # Determine which hand has data and process it
            active_hand_data = None
            hand_type = None
            
            if right_hand_data and right_hand_data.get('landmarks'):
                active_hand_data = right_hand_data
                hand_type = 'right'
            elif left_hand_data and left_hand_data.get('landmarks'):
                active_hand_data = left_hand_data  
                hand_type = 'left'
            
            if active_hand_data:
                landmarks = np.array(active_hand_data['landmarks'])[:, :3]  # x, y, z only
                
                if len(landmarks) >= 21:
                    # Normalize relative to wrist
                    wrist = landmarks[0]
                    normalized = landmarks - wrist
                    
                    # Scale by middle finger length for consistency
                    middle_finger_length = np.linalg.norm(normalized[12]) + 1e-8
                    normalized = normalized / middle_finger_length
                    
                    # Key hand shape features
                    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                    
                    # 1. Normalized landmark positions (21 points * 3 coords = 63 features)
                    feature_vector.extend(normalized.flatten())
                    
                    # 2. Finger curl ratios (5 features)
                    for tip_idx in fingertips:
                        # Distance from wrist to tip vs expected extended distance
                        tip_distance = np.linalg.norm(normalized[tip_idx])
                        feature_vector.append(tip_distance)
                    
                    # 3. Finger spreads (4 features - between adjacent fingers)
                    for i in range(len(fingertips) - 1):
                        spread = np.linalg.norm(normalized[fingertips[i]] - normalized[fingertips[i+1]])
                        feature_vector.append(spread)
                    
                    # 4. Hand openness indicators (3 features)
                    # Thumb position relative to index
                    thumb_index_dist = np.linalg.norm(normalized[4] - normalized[8])
                    # Average fingertip distance from palm center
                    palm_center = np.mean([normalized[5], normalized[9], normalized[13], normalized[17]], axis=0)  # MCP joints
                    avg_tip_distance = np.mean([np.linalg.norm(normalized[tip] - palm_center) for tip in fingertips])
                    # Hand span (thumb to pinky)
                    hand_span = np.linalg.norm(normalized[4] - normalized[20])
                    
                    feature_vector.extend([thumb_index_dist, avg_tip_distance, hand_span])
                    
                    # 5. Hand type indicator (1 feature)
                    feature_vector.append(1.0 if hand_type == 'right' else -1.0)
                    
                else:
                    # Not enough landmarks
                    feature_vector.extend([0] * 76)  # 63 + 5 + 4 + 3 + 1
            else:
                # No hand detected
                feature_vector.extend([0] * 76)
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"⚠️  Error extracting features from sample: {e}")
            return None
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train the recognition model."""
        try:
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Use KNN classifier (good for small datasets)
            self.model = KNeighborsClassifier(n_neighbors=min(3, len(X)), weights='distance')
            self.model.fit(X_scaled, y)
            
            # Create label mapping
            unique_labels = np.unique(y)
            self.label_encoder = {i: label for i, label in enumerate(unique_labels)}
            
            print(f"✅ Model trained successfully!")
            print(f"   • Algorithm: K-Nearest Neighbors")
            print(f"   • Training samples: {len(X)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def recognize_from_detection(self, detection_data: Dict) -> Tuple[Optional[str], float]:
        """Recognize letter from detection data."""
        if self.model is None:
            return None, 0.0
        
        try:
            # Check if any hand is detected first
            hands_data = detection_data.get('hands', {})
            left_hand = hands_data.get('left_hand')
            right_hand = hands_data.get('right_hand')
            
            # If no hands detected, return None
            if not ((left_hand and left_hand.get('landmarks')) or (right_hand and right_hand.get('landmarks'))):
                return None, 0.0
            
            # Extract features directly from detection data
            feature_vector = self._extract_features_from_current_frame(detection_data)
            
            if feature_vector is None or np.all(feature_vector == 0):
                return None, 0.0
            
            # Normalize and predict
            X_scaled = self.scaler.transform([feature_vector])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X_scaled)[0]
            predicted_class = self.model.predict(X_scaled)[0]
            confidence = np.max(probabilities)
            
            # Enhanced confidence checking for single-class models
            # Calculate similarity to training data
            similarity_score = self._calculate_similarity_to_training_data(feature_vector)
            
            # Combine model confidence with similarity score
            final_confidence = confidence * similarity_score
            
            # Apply stricter threshold for single-class models
            strict_threshold = 0.7  # Higher threshold for better accuracy
            
            if final_confidence >= strict_threshold and similarity_score > 0.3:
                return predicted_class, final_confidence
            else:
                return None, final_confidence
                
        except Exception as e:
            print(f"⚠️  Recognition error: {e}")
            return None, 0.0
    
    def _calculate_similarity_to_training_data(self, current_features: np.ndarray) -> float:
        """Calculate how similar current features are to training data."""
        try:
            if not hasattr(self, 'training_features'):
                return 0.5  # Default similarity if no training data available
            
            # Calculate average distance to all training samples
            distances = []
            for train_features in self.training_features:
                # Normalize both vectors
                current_norm = current_features / (np.linalg.norm(current_features) + 1e-8)
                train_norm = train_features / (np.linalg.norm(train_features) + 1e-8)
                
                # Calculate cosine similarity
                similarity = np.dot(current_norm, train_norm)
                distances.append(max(0, similarity))  # Keep only positive similarities
            
            if len(distances) == 0:
                return 0.0
            
            # Return average similarity
            avg_similarity = np.mean(distances)
            return avg_similarity
            
        except Exception as e:
            print(f"⚠️  Similarity calculation error: {e}")
            return 0.0
    
    def _extract_features_from_current_frame(self, features: Dict) -> Optional[np.ndarray]:
        """Extract hand-focused feature vector from current frame (same as training)."""
        try:
            # Get hands data directly from detection (bypass feature extractor complexity)
            detection_data = features  # This is actually the raw detection data
            hands_data = detection_data.get('hands', {})
            
            feature_vector = []
            
            # Check for active hand (prioritize right, then left)
            left_hand_data = hands_data.get('left_hand')
            right_hand_data = hands_data.get('right_hand')
            
            active_hand_data = None
            hand_type = None
            
            if right_hand_data and right_hand_data.get('landmarks'):
                active_hand_data = right_hand_data
                hand_type = 'right'
            elif left_hand_data and left_hand_data.get('landmarks'):
                active_hand_data = left_hand_data
                hand_type = 'left'
            
            if active_hand_data:
                landmarks = np.array(active_hand_data['landmarks'])[:, :3]
                
                if len(landmarks) >= 21:
                    # Same processing as training data
                    wrist = landmarks[0]
                    normalized = landmarks - wrist
                    
                    # Scale by middle finger length
                    middle_finger_length = np.linalg.norm(normalized[12]) + 1e-8
                    normalized = normalized / middle_finger_length
                    
                    fingertips = [4, 8, 12, 16, 20]
                    
                    # 1. Normalized landmarks (63 features)
                    feature_vector.extend(normalized.flatten())
                    
                    # 2. Finger curl ratios (5 features)
                    for tip_idx in fingertips:
                        tip_distance = np.linalg.norm(normalized[tip_idx])
                        feature_vector.append(tip_distance)
                    
                    # 3. Finger spreads (4 features)
                    for i in range(len(fingertips) - 1):
                        spread = np.linalg.norm(normalized[fingertips[i]] - normalized[fingertips[i+1]])
                        feature_vector.append(spread)
                    
                    # 4. Hand openness (3 features)
                    thumb_index_dist = np.linalg.norm(normalized[4] - normalized[8])
                    palm_center = np.mean([normalized[5], normalized[9], normalized[13], normalized[17]], axis=0)
                    avg_tip_distance = np.mean([np.linalg.norm(normalized[tip] - palm_center) for tip in fingertips])
                    hand_span = np.linalg.norm(normalized[4] - normalized[20])
                    
                    feature_vector.extend([thumb_index_dist, avg_tip_distance, hand_span])
                    
                    # 5. Hand type (1 feature)
                    feature_vector.append(1.0 if hand_type == 'right' else -1.0)
                    
                else:
                    feature_vector.extend([0] * 76)
            else:
                feature_vector.extend([0] * 76)
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"⚠️  Error extracting current frame features: {e}")
            return None
    
    def update_recognition_history(self, prediction: Optional[str], confidence: float):
        """Update recognition history for stability."""
        self.recognition_history.append((prediction, confidence))
        
        # Keep only recent history
        if len(self.recognition_history) > self.history_size:
            self.recognition_history.pop(0)
    
    def get_stable_prediction(self) -> Tuple[Optional[str], float]:
        """Get stable prediction from history."""
        if len(self.recognition_history) < 2:
            return None, 0.0
        
        # Count predictions
        predictions = {}
        total_confidence = 0
        valid_count = 0
        
        for pred, conf in self.recognition_history[-3:]:  # Last 3 frames
            if pred is not None:
                predictions[pred] = predictions.get(pred, 0) + 1
                total_confidence += conf
                valid_count += 1
        
        if valid_count == 0:
            return None, 0.0
        
        # Get most common prediction
        if predictions:
            most_common = max(predictions, key=predictions.get)
            avg_confidence = total_confidence / valid_count
            
            # Require at least 2 out of 3 recent frames to agree
            if predictions[most_common] >= 2:
                return most_common, avg_confidence
        
        return None, 0.0
    
    def draw_recognition_info(self, frame: np.ndarray, prediction: Optional[str], 
                            confidence: float, fps: float) -> np.ndarray:
        """Draw recognition information on frame."""
        height, width = frame.shape[:2]
        
        # Background panel for info
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, "LESSA Alphabet Recognizer", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Trained letters
        letters_text = f"Trained: {', '.join(sorted(self.trained_letters))}"
        cv2.putText(frame, letters_text, (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Current prediction
        if prediction:
            pred_text = f"Recognized: {prediction}"
            conf_text = f"Confidence: {confidence:.3f}"
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 200, 200)
            
            cv2.putText(frame, pred_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, conf_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Large letter display
            cv2.putText(frame, prediction, (width - 100, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
        else:
            status_text = "No letter detected"
            if confidence > 0:
                status_text = f"Unknown sign (conf: {confidence:.3f})"
            
            cv2.putText(frame, status_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 100, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Controls:",
            "L - Toggle landmarks",
            "I - Toggle info",
            "R - Reload training data",
            "C - Clear feature cache", 
            "Q - Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - 120 + (i * 20)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def run_recognition(self):
        """Run real-time alphabet recognition."""
        if self.model is None:
            print("❌ No trained model available. Please collect training data first.")
            return
        
        print("🚀 Starting LESSA Alphabet Recognition...")
        print(f"📊 Ready to recognize: {', '.join(sorted(self.trained_letters))}")
        print("\n🎯 Position your hand in front of the camera and sign letters!")
        
        # Initialize camera
        try:
            camera_manager = CameraManager()
            cameras = camera_manager.detect_cameras()
            
            if not cameras:
                print("❌ No cameras detected!")
                return
            
            best_camera = camera_manager.get_best_camera()
            cap = cv2.VideoCapture(best_camera.device_id)
            
            if not cap.isOpened():
                print("❌ Could not open camera!")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return
        
        # Recognition loop
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect holistic features
                annotated_frame, detection_data = self.detector.detect_holistic(frame)
                
                # Recognize letter
                prediction, confidence = self.recognize_from_detection(detection_data)
                
                # Update history and get stable prediction
                self.update_recognition_history(prediction, confidence)
                stable_pred, stable_conf = self.get_stable_prediction()
                
                # Use annotated frame if landmarks enabled
                if self.show_landmarks:
                    frame = annotated_frame  # Use the annotated frame from detection
                
                # Draw recognition info
                if self.show_prediction_info:
                    frame = self.draw_recognition_info(frame, stable_pred, stable_conf, current_fps)
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()
                
                # Display frame
                cv2.imshow("LESSA Alphabet Recognizer", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('i'):
                    self.show_prediction_info = not self.show_prediction_info
                    print(f"Info display: {'ON' if self.show_prediction_info else 'OFF'}")
                elif key == ord('r'):
                    print("🔄 Reloading training data...")
                    if self.load_training_data():
                        print("✅ Training data reloaded successfully!")
                    else:
                        print("❌ Failed to reload training data")
                elif key == ord('c'):
                    print("🗑️  Clearing feature cache...")
                    self.feature_cache.clear_cache()
                    print("   Cache cleared. Next reload will rebuild cache.")
        
        except KeyboardInterrupt:
            print("\n👋 Recognition stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("📹 Camera released")

def run_alphabet_recognizer():
    """Entry point for the alphabet recognizer."""
    recognizer = AlphabetRecognizer()
    recognizer.run_recognition()

if __name__ == "__main__":
    run_alphabet_recognizer()