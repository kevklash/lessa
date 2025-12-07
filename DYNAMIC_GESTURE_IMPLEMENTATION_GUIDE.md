# LESSA Dynamic Gesture Implementation Guide

## Technical Implementation Plan for Dynamic Letter Recognition

### Overview
This guide provides detailed implementation steps for enhancing the LESSA system to support dynamic gesture recognition, specifically focusing on letters J and Z that require movement patterns.

## 1. Enhanced Data Collection System

### 1.1 Dynamic Letter Collector Enhancement

```python
"""
Enhanced alphabet collector for dynamic gesture collection
Extends existing AlphabetCollector with temporal sequence capture
"""

import cv2
import numpy as np
import time
import json
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class DynamicLetterCollector:
    """Enhanced collector for dynamic sign language letters."""
    
    def __init__(self, base_collector):
        self.base_collector = base_collector
        self.dynamic_letters = {'J', 'Z'}  # Start with these two
        self.sequence_length = 30  # frames for dynamic gestures
        self.frame_rate = 30  # target FPS
        
        # Temporal collection parameters
        self.motion_threshold = 0.02  # minimum motion to start collection
        self.idle_threshold = 0.005   # motion below this ends collection
        self.min_gesture_frames = 10  # minimum frames for valid gesture
        self.max_gesture_frames = 90  # maximum frames for valid gesture
        
        # Collection state
        self.collection_state = 'idle'  # idle, detecting, collecting, complete
        self.temporal_buffer = deque(maxlen=self.max_gesture_frames)
        self.motion_history = deque(maxlen=10)
        
        print("🎬 Dynamic Letter Collector initialized!")
        print(f"📝 Dynamic letters: {', '.join(self.dynamic_letters)}")
    
    def is_dynamic_letter(self, letter: str) -> bool:
        """Check if letter requires dynamic collection."""
        return letter in self.dynamic_letters
    
    def calculate_hand_motion(self, current_landmarks, previous_landmarks):
        """Calculate motion magnitude between consecutive frames."""
        if previous_landmarks is None:
            return 0.0
        
        # Calculate Euclidean distance for all hand landmarks
        motion = 0.0
        landmark_count = 0
        
        for i in range(21):  # 21 hand landmarks
            if (current_landmarks.landmark[i].visibility > 0.5 and 
                previous_landmarks.landmark[i].visibility > 0.5):
                
                dx = current_landmarks.landmark[i].x - previous_landmarks.landmark[i].x
                dy = current_landmarks.landmark[i].y - previous_landmarks.landmark[i].y
                motion += np.sqrt(dx*dx + dy*dy)
                landmark_count += 1
        
        return motion / max(landmark_count, 1)
    
    def update_collection_state(self, motion_magnitude: float):
        """Update collection state based on motion."""
        self.motion_history.append(motion_magnitude)
        avg_motion = np.mean(self.motion_history)
        
        if self.collection_state == 'idle':
            if avg_motion > self.motion_threshold:
                self.collection_state = 'detecting'
                self.temporal_buffer.clear()
                print("🔍 Motion detected - starting collection")
        
        elif self.collection_state == 'detecting':
            if avg_motion > self.motion_threshold:
                self.collection_state = 'collecting'
                print("📹 Collecting gesture sequence...")
            elif avg_motion < self.idle_threshold:
                self.collection_state = 'idle'
        
        elif self.collection_state == 'collecting':
            if avg_motion < self.idle_threshold:
                if len(self.temporal_buffer) >= self.min_gesture_frames:
                    self.collection_state = 'complete'
                    print(f"✅ Gesture sequence complete ({len(self.temporal_buffer)} frames)")
                else:
                    self.collection_state = 'idle'
                    print("❌ Sequence too short - restarting")
    
    def collect_dynamic_sample(self, letter: str, camera, detector, feature_extractor):
        """Collect a dynamic gesture sample."""
        if not self.is_dynamic_letter(letter):
            # Fall back to static collection
            return self.base_collector.collect_sample(letter, camera, detector, feature_extractor)
        
        print(f"\n🎬 Collecting dynamic sample for letter: {letter}")
        print("📋 Instructions:")
        print(f"   - Perform the {letter} gesture naturally")
        print("   - Wait for motion detection")
        print("   - Complete the gesture smoothly")
        print("   - Press 's' to save, 'r' to retry, 'q' to quit")
        
        collected_sample = None
        previous_landmarks = None
        
        while collected_sample is None:
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect holistic landmarks
            annotated_frame, detection_data = detector.detect_holistic(frame)
            
            # Extract hand landmarks if available
            if detection_data.get('right_hand_landmarks'):
                current_landmarks = detection_data['right_hand_landmarks']
                
                # Calculate motion if we have previous frame
                if previous_landmarks is not None:
                    motion = self.calculate_hand_motion(current_landmarks, previous_landmarks)
                    self.update_collection_state(motion)
                
                # Add frame to buffer if collecting
                if self.collection_state in ['detecting', 'collecting']:
                    timestamp = time.time()
                    features = feature_extractor.extract_features(detection_data)
                    
                    frame_data = {
                        'timestamp': timestamp,
                        'landmarks': detection_data,
                        'features': features,
                        'motion': motion if previous_landmarks else 0.0
                    }
                    
                    self.temporal_buffer.append(frame_data)
                
                # Check if collection is complete
                if self.collection_state == 'complete':
                    collected_sample = {
                        'letter': letter,
                        'sequence': list(self.temporal_buffer),
                        'timestamp': datetime.now().isoformat(),
                        'sequence_length': len(self.temporal_buffer),
                        'type': 'dynamic'
                    }
                    self.collection_state = 'idle'
                
                previous_landmarks = current_landmarks
            
            # Display collection status
            self.draw_collection_interface(annotated_frame, letter)
            
            # Show frame
            cv2.imshow('Dynamic Letter Collection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and collected_sample:
                return collected_sample
            elif key == ord('r'):
                self.collection_state = 'idle'
                self.temporal_buffer.clear()
                print("🔄 Restarting collection...")
            elif key == ord('q'):
                return None
        
        return collected_sample
    
    def draw_collection_interface(self, frame, letter):
        """Draw collection status on frame."""
        height, width = frame.shape[:2]
        
        # Status panel
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, f"Collecting Dynamic Letter: {letter}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # State
        state_colors = {
            'idle': (128, 128, 128),
            'detecting': (0, 255, 255),
            'collecting': (0, 255, 0),
            'complete': (0, 128, 255)
        }
        
        state_color = state_colors.get(self.collection_state, (255, 255, 255))
        cv2.putText(frame, f"State: {self.collection_state.upper()}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # Buffer status
        buffer_text = f"Frames: {len(self.temporal_buffer)}/{self.max_gesture_frames}"
        cv2.putText(frame, buffer_text, 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Motion indicator
        if self.motion_history:
            motion_level = int(np.mean(self.motion_history) * 1000)
            motion_text = f"Motion: {motion_level}"
            cv2.putText(frame, motion_text, 
                       (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


## 2. Temporal Feature Extraction

### 2.1 Enhanced Feature Extractor

class TemporalFeatureExtractor:
    """Extract temporal features from gesture sequences."""
    
    def __init__(self):
        self.feature_dim = 84  # Based on enhanced feature set
        
    def extract_sequence_features(self, gesture_sequence: List[Dict]) -> np.ndarray:
        """Extract features from entire gesture sequence."""
        if not gesture_sequence:
            return np.zeros((1, self.feature_dim))
        
        features = []
        
        for i, frame_data in enumerate(gesture_sequence):
            frame_features = self.extract_frame_features(
                frame_data, 
                gesture_sequence[i-1] if i > 0 else None
            )
            features.append(frame_features)
        
        # Normalize sequence length to 30 frames
        features = self.normalize_sequence_length(features, target_length=30)
        
        return np.array(features)
    
    def extract_frame_features(self, frame_data: Dict, previous_frame: Optional[Dict] = None) -> List[float]:
        """Extract features from a single frame."""
        landmarks = frame_data['landmarks']
        features = []
        
        # 1. Hand landmark positions (42 features: 21 landmarks × 2 coordinates)
        if landmarks.get('right_hand_landmarks'):
            hand_landmarks = landmarks['right_hand_landmarks']
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y])
        else:
            features.extend([0.0] * 42)  # Zero padding if no hand detected
        
        # 2. Hand velocity features (42 features if previous frame available)
        if previous_frame and landmarks.get('right_hand_landmarks'):
            prev_landmarks = previous_frame['landmarks'].get('right_hand_landmarks')
            if prev_landmarks:
                velocities = self.calculate_velocities(
                    landmarks['right_hand_landmarks'], 
                    prev_landmarks
                )
                features.extend(velocities)
            else:
                features.extend([0.0] * 42)
        else:
            features.extend([0.0] * 42)
        
        # Ensure we have exactly the expected number of features
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return features[:self.feature_dim]
    
    def calculate_velocities(self, current_landmarks, previous_landmarks) -> List[float]:
        """Calculate velocity for each landmark."""
        velocities = []
        
        for i in range(21):  # 21 hand landmarks
            if (i < len(current_landmarks.landmark) and 
                i < len(previous_landmarks.landmark)):
                
                curr = current_landmarks.landmark[i]
                prev = previous_landmarks.landmark[i]
                
                vx = curr.x - prev.x
                vy = curr.y - prev.y
                
                velocities.extend([vx, vy])
            else:
                velocities.extend([0.0, 0.0])
        
        return velocities
    
    def normalize_sequence_length(self, features: List[List[float]], target_length: int) -> List[List[float]]:
        """Normalize sequence to target length using interpolation."""
        if len(features) == target_length:
            return features
        
        features_array = np.array(features)
        
        if len(features) < target_length:
            # Interpolate to expand
            old_indices = np.linspace(0, len(features) - 1, len(features))
            new_indices = np.linspace(0, len(features) - 1, target_length)
            
            interpolated = []
            for i in range(features_array.shape[1]):
                interpolated_feature = np.interp(new_indices, old_indices, features_array[:, i])
                interpolated.append(interpolated_feature)
            
            return np.array(interpolated).T.tolist()
        else:
            # Downsample
            step = len(features) / target_length
            indices = [int(i * step) for i in range(target_length)]
            return [features[i] for i in indices]


## 3. Dynamic Gesture Classifier

### 3.1 LSTM-Based Classifier

class DynamicGestureClassifier:
    """LSTM-based classifier for dynamic gestures."""
    
    def __init__(self, sequence_length: int = 30, feature_dim: int = 84):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.label_encoder = {'J': 0, 'Z': 1}
        self.reverse_label_encoder = {0: 'J', 1: 'Z'}
        
    def build_model(self):
        """Build LSTM model for dynamic gesture classification."""
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            # First LSTM layer
            tf.keras.layers.LSTM(
                128, 
                return_sequences=True, 
                input_shape=(self.sequence_length, self.feature_dim),
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            
            # Second LSTM layer
            tf.keras.layers.LSTM(
                64, 
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            
            # Dense layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.label_encoder), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_training_data(self, collected_samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from collected samples."""
        feature_extractor = TemporalFeatureExtractor()
        
        X, y = [], []
        
        for sample in collected_samples:
            if sample['type'] == 'dynamic' and sample['letter'] in self.label_encoder:
                # Extract features from sequence
                features = feature_extractor.extract_sequence_features(sample['sequence'])
                
                # Create label
                label = self.label_encoder[sample['letter']]
                label_onehot = tf.keras.utils.to_categorical(label, len(self.label_encoder))
                
                X.append(features)
                y.append(label_onehot)
        
        return np.array(X), np.array(y)
    
    def train(self, training_samples: List[Dict], validation_split: float = 0.2):
        """Train the dynamic gesture classifier."""
        import tensorflow as tf
        
        # Prepare data
        X, y = self.prepare_training_data(training_samples)
        
        if len(X) == 0:
            raise ValueError("No training samples available")
        
        print(f"Training with {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, gesture_sequence: List[Dict]) -> Tuple[str, float]:
        """Predict gesture from sequence."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Extract features
        feature_extractor = TemporalFeatureExtractor()
        features = feature_extractor.extract_sequence_features(gesture_sequence)
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        predicted_letter = self.reverse_label_encoder[predicted_class]
        
        return predicted_letter, confidence
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


## 4. Integration with Existing LESSA

### 4.1 Enhanced Alphabet Recognizer

class EnhancedAlphabetRecognizer:
    """Enhanced recognizer supporting both static and dynamic gestures."""
    
    def __init__(self, static_recognizer):
        self.static_recognizer = static_recognizer
        self.dynamic_classifier = DynamicGestureClassifier()
        self.feature_extractor = TemporalFeatureExtractor()
        
        # Temporal processing
        self.temporal_buffer = deque(maxlen=30)
        self.motion_history = deque(maxlen=10)
        self.gesture_state = 'idle'  # idle, detecting, collecting, analyzing
        
        # Thresholds
        self.motion_start_threshold = 0.02
        self.motion_end_threshold = 0.005
        self.min_sequence_length = 10
        
        # Dynamic letters
        self.dynamic_letters = {'J', 'Z'}
        
        print("🚀 Enhanced Alphabet Recognizer initialized!")
        print(f"📝 Supporting dynamic letters: {', '.join(self.dynamic_letters)}")
    
    def load_dynamic_model(self, model_path: str):
        """Load pre-trained dynamic gesture model."""
        try:
            self.dynamic_classifier.load_model(model_path)
            print(f"✅ Dynamic model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load dynamic model: {e}")
            return False
    
    def calculate_motion(self, current_landmarks, previous_landmarks):
        """Calculate motion magnitude between frames."""
        if not current_landmarks or not previous_landmarks:
            return 0.0
        
        # Use the same motion calculation as collector
        motion = 0.0
        landmark_count = 0
        
        for i in range(21):
            if (i < len(current_landmarks.landmark) and 
                i < len(previous_landmarks.landmark)):
                
                curr = current_landmarks.landmark[i]
                prev = previous_landmarks.landmark[i]
                
                if curr.visibility > 0.5 and prev.visibility > 0.5:
                    dx = curr.x - prev.x
                    dy = curr.y - prev.y
                    motion += np.sqrt(dx*dx + dy*dy)
                    landmark_count += 1
        
        return motion / max(landmark_count, 1)
    
    def update_gesture_state(self):
        """Update gesture recognition state."""
        if len(self.motion_history) < 3:
            return
        
        avg_motion = np.mean(list(self.motion_history)[-3:])  # Recent motion
        
        if self.gesture_state == 'idle':
            if avg_motion > self.motion_start_threshold:
                self.gesture_state = 'detecting'
                self.temporal_buffer.clear()
        
        elif self.gesture_state == 'detecting':
            if avg_motion > self.motion_start_threshold:
                self.gesture_state = 'collecting'
            elif avg_motion < self.motion_end_threshold:
                self.gesture_state = 'idle'
        
        elif self.gesture_state == 'collecting':
            if avg_motion < self.motion_end_threshold:
                if len(self.temporal_buffer) >= self.min_sequence_length:
                    self.gesture_state = 'analyzing'
                else:
                    self.gesture_state = 'idle'
    
    def process_frame(self, frame, landmarks_data):
        """Process frame with temporal awareness."""
        current_time = time.time()
        result = None
        
        # Extract hand landmarks
        hand_landmarks = landmarks_data.get('right_hand_landmarks')
        
        if hand_landmarks:
            # Calculate motion if we have previous frame
            motion = 0.0
            if len(self.temporal_buffer) > 0:
                prev_data = self.temporal_buffer[-1]
                prev_landmarks = prev_data['landmarks'].get('right_hand_landmarks')
                if prev_landmarks:
                    motion = self.calculate_motion(hand_landmarks, prev_landmarks)
            
            # Update motion history
            self.motion_history.append(motion)
            
            # Add to temporal buffer
            frame_data = {
                'timestamp': current_time,
                'landmarks': landmarks_data,
                'motion': motion
            }
            self.temporal_buffer.append(frame_data)
            
            # Update state
            self.update_gesture_state()
            
            # Process based on current state
            if self.gesture_state == 'analyzing':
                result = self.analyze_gesture_sequence()
                self.gesture_state = 'idle'  # Reset after analysis
            elif self.gesture_state in ['detecting', 'collecting']:
                result = {
                    'letter': '?',
                    'confidence': 0.0,
                    'status': f'Collecting gesture... ({len(self.temporal_buffer)} frames)',
                    'type': 'dynamic_collecting'
                }
            else:
                # Static gesture recognition
                result = self.static_recognizer.recognize_single_frame(landmarks_data)
        
        return result
    
    def analyze_gesture_sequence(self):
        """Analyze collected gesture sequence."""
        if len(self.temporal_buffer) < self.min_sequence_length:
            return {
                'letter': '?',
                'confidence': 0.0,
                'status': 'Sequence too short',
                'type': 'error'
            }
        
        try:
            # Convert buffer to sequence format
            sequence = list(self.temporal_buffer)
            
            # Try dynamic classification
            if self.dynamic_classifier.model is not None:
                predicted_letter, confidence = self.dynamic_classifier.predict(sequence)
                
                if confidence > 0.7:  # High confidence threshold for dynamic
                    return {
                        'letter': predicted_letter,
                        'confidence': confidence,
                        'status': f'Dynamic gesture recognized',
                        'type': 'dynamic',
                        'sequence_length': len(sequence)
                    }
            
            # Fall back to static recognition using last frame
            last_frame_landmarks = sequence[-1]['landmarks']
            static_result = self.static_recognizer.recognize_single_frame(last_frame_landmarks)
            
            return {
                'letter': static_result.get('letter', '?'),
                'confidence': static_result.get('confidence', 0.0),
                'status': 'Fallback to static recognition',
                'type': 'static_fallback'
            }
            
        except Exception as e:
            print(f"Error analyzing gesture sequence: {e}")
            return {
                'letter': '?',
                'confidence': 0.0,
                'status': f'Analysis error: {e}',
                'type': 'error'
            }
    
    def get_gesture_state_info(self):
        """Get current gesture recognition state information."""
        return {
            'state': self.gesture_state,
            'buffer_length': len(self.temporal_buffer),
            'recent_motion': np.mean(list(self.motion_history)[-3:]) if len(self.motion_history) >= 3 else 0,
            'is_dynamic_mode': self.gesture_state != 'idle'
        }


## 5. Training Pipeline

### 5.1 Complete Training Script

def train_dynamic_gesture_model(data_file: str = "lessa_alphabet_data.json"):
    """Complete training pipeline for dynamic gestures."""
    
    print("🚀 Starting Dynamic Gesture Model Training")
    
    # Load collected data
    try:
        with open(data_file, 'r') as f:
            all_data = json.load(f)
        print(f"📊 Loaded data from {data_file}")
    except FileNotFoundError:
        print(f"❌ Data file {data_file} not found")
        return None
    
    # Filter dynamic gesture samples
    dynamic_samples = []
    for letter, samples in all_data.items():
        if letter in ['J', 'Z']:  # Dynamic letters
            for sample in samples:
                if isinstance(sample, dict) and sample.get('type') == 'dynamic':
                    dynamic_samples.append(sample)
    
    print(f"📝 Found {len(dynamic_samples)} dynamic samples")
    
    if len(dynamic_samples) < 10:
        print("❌ Insufficient training data. Need at least 10 samples.")
        return None
    
    # Initialize classifier
    classifier = DynamicGestureClassifier()
    
    # Train model
    print("🎯 Training model...")
    try:
        history = classifier.train(dynamic_samples)
        
        # Save model
        model_path = "dynamic_gesture_model.h5"
        classifier.save_model(model_path)
        
        print("✅ Training completed successfully!")
        print(f"💾 Model saved to {model_path}")
        
        return classifier
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


## 6. Usage Example

### 6.1 Complete Integration Example

def main():
    """Example usage of enhanced LESSA system."""
    
    # Initialize components
    from src.utils.enhanced_camera import CameraManager
    from src.detection.holistic_detector import HolisticDetector
    from src.detection.holistic_feature_extractor import HolisticFeatureExtractor
    
    # Initialize camera and detection
    camera_manager = CameraManager()
    camera = camera_manager.get_best_camera()
    detector = HolisticDetector()
    feature_extractor = HolisticFeatureExtractor()
    
    # Initialize enhanced recognizer
    base_recognizer = AlphabetRecognizer()  # Your existing recognizer
    enhanced_recognizer = EnhancedAlphabetRecognizer(base_recognizer)
    
    # Load dynamic model if available
    enhanced_recognizer.load_dynamic_model("dynamic_gesture_model.h5")
    
    print("🚀 Enhanced LESSA system ready!")
    print("📝 Supports both static and dynamic letter recognition")
    print("🎬 Dynamic letters: J, Z")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect landmarks
            annotated_frame, landmarks_data = detector.detect_holistic(frame)
            
            # Enhanced recognition
            result = enhanced_recognizer.process_frame(frame, landmarks_data)
            
            # Display results
            if result:
                draw_recognition_results(annotated_frame, result, enhanced_recognizer)
            
            # Show frame
            cv2.imshow('Enhanced LESSA - Dynamic Gesture Recognition', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()

def draw_recognition_results(frame, result, recognizer):
    """Draw recognition results on frame."""
    height, width = frame.shape[:2]
    
    # Status panel
    panel_height = 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_pos = height - panel_height + 30
    
    # Recognition result
    letter = result.get('letter', '?')
    confidence = result.get('confidence', 0.0)
    result_type = result.get('type', 'unknown')
    
    # Color based on type
    colors = {
        'dynamic': (0, 255, 0),      # Green for dynamic
        'static': (255, 255, 0),     # Yellow for static  
        'dynamic_collecting': (0, 255, 255),  # Cyan for collecting
        'static_fallback': (255, 128, 0),    # Orange for fallback
        'error': (0, 0, 255)         # Red for error
    }
    
    color = colors.get(result_type, (255, 255, 255))
    
    # Main result
    cv2.putText(frame, f"Letter: {letter} ({confidence:.2f})", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Status
    status = result.get('status', '')
    cv2.putText(frame, f"Status: {status}", 
               (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Gesture state info
    state_info = recognizer.get_gesture_state_info()
    state_text = f"Mode: {state_info['state']} | Buffer: {state_info['buffer_length']}"
    cv2.putText(frame, state_text, 
               (10, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

if __name__ == "__main__":
    main()
```

## Summary

This implementation guide provides:

1. **Enhanced Data Collection**: Temporal sequence capture for dynamic gestures
2. **Feature Extraction**: Spatial and temporal feature extraction pipeline  
3. **LSTM Classifier**: Deep learning model for dynamic gesture recognition
4. **Seamless Integration**: Works with existing LESSA architecture
5. **Complete Pipeline**: From data collection to real-time recognition

### Next Steps:

1. **Collect Training Data**: Use the enhanced collector to gather J and Z samples
2. **Train Model**: Run the training pipeline with collected data
3. **Test Integration**: Verify the enhanced recognizer works correctly
4. **Optimize Performance**: Fine-tune for real-time operation
5. **Expand Coverage**: Add more dynamic gestures as needed

The system maintains backward compatibility while adding powerful dynamic gesture recognition capabilities, making LESSA more complete for sign language recognition.