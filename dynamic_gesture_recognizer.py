"""
Dynamic gesture recognizer for movement-based LESSA letters (J, Z).
Uses LSTM neural network for temporal sequence classification.
"""

import numpy as np
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

# Suppress TensorFlow warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

from src.detection.temporal_feature_extractor import TemporalFeatureExtractor


class DynamicGestureRecognizer:
    """LSTM-based dynamic gesture recognizer for LESSA letters J and Z."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize dynamic gesture recognizer."""
        self.feature_extractor = TemporalFeatureExtractor()
        self.dynamic_letters = ['J', 'Z']
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training parameters
        self.sequence_length = 30
        self.batch_size = 16
        self.epochs = 100
        self.validation_split = 0.2
        
        # Recognition parameters
        self.confidence_threshold = 0.7
        self.recognition_history = []
        self.history_size = 5
        
        # File paths
        self.model_dir = Path("models/dynamic")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "dynamic_gesture_model.h5"
        self.scaler_path = self.model_dir / "dynamic_scaler.pkl"
        self.encoder_path = self.model_dir / "dynamic_encoder.pkl"
        self.data_file = Path("lessa_dynamic_data.json")
        
        # Load existing model if specified
        if model_path:
            self.load_model(model_path)
        elif self.model_path.exists():
            print(f"📁 Loading existing model from {self.model_path}")
            self.load_model()
            
    def load_training_data(self) -> bool:
        """Load and prepare training data."""
        try:
            if not self.data_file.exists():
                print(f"❌ No training data found at {self.data_file}")
                return False
                
            print(f"📊 Loading dynamic gesture training data...")
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                
            # Extract features and labels
            features_list = []
            labels_list = []
            
            total_samples = 0
            for letter in self.dynamic_letters:
                if letter in data:
                    letter_samples = data[letter]
                    print(f"   • {letter}: {len(letter_samples)} samples")
                    
                    for sample in letter_samples:
                        # Extract features from sequence
                        gesture_sequence = sample.get('landmarks_sequence', [])
                        if len(gesture_sequence) >= 3:  # Minimum frames needed
                            feature_data = self.feature_extractor.extract_features(gesture_sequence)
                            
                            if feature_data and feature_data['feature_vector'].size > 0:
                                features_list.append(feature_data['feature_vector'])
                                labels_list.append(letter)
                                total_samples += 1
                                
            if total_samples == 0:
                print("❌ No valid samples found in training data")
                return False
                
            print(f"✅ Loaded {total_samples} valid samples")
            
            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(labels_list)
            
            print(f"📈 Feature matrix shape: {X.shape}")
            
            # Store for model training
            self.X_train = X
            self.y_train = y
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
            
    def _create_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Create LSTM model architecture."""
        try:
            model = Sequential([
                # Reshape for LSTM (batch_size, sequence_length, features)
                # For now, we'll treat the flattened features as a sequence of 1
                Dense(512, activation='relu', input_shape=input_shape),
                Dropout(0.3),
                BatchNormalization(),
                
                # Reshape for LSTM processing
                tf.keras.layers.Reshape((32, -1)),  # 32 time steps
                
                # LSTM layers
                LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
                LSTM(64, dropout=0.3, recurrent_dropout=0.3),
                
                # Dense layers
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(len(self.dynamic_letters), activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            return None
            
    def train_model(self) -> bool:
        """Train the LSTM model on collected data."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available for training")
            return False
            
        try:
            # Load training data
            if not self.load_training_data():
                return False
                
            X, y = self.X_train, self.y_train
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded, 
                test_size=self.validation_split, 
                random_state=42,
                stratify=y_encoded
            )
            
            print(f"🔄 Training model...")
            print(f"   • Training samples: {len(X_train)}")
            print(f"   • Validation samples: {len(X_val)}")
            print(f"   • Feature dimensions: {X_scaled.shape[1]}")
            
            # Create model
            self.model = self._create_model(input_shape=(X_scaled.shape[1],))
            
            if self.model is None:
                return False
                
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            print(f"\n✅ Training completed!")
            print(f"   • Validation accuracy: {val_accuracy:.3f}")
            print(f"   • Validation loss: {val_loss:.3f}")
            
            # Generate classification report
            y_pred = self.model.predict(X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            target_names = self.label_encoder.classes_
            report = classification_report(y_val, y_pred_classes, 
                                         target_names=target_names,
                                         zero_division=0)
            print("\n📊 Classification Report:")
            print(report)
            
            # Save model and components
            self.save_model()
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
            
    def save_model(self):
        """Save trained model and preprocessing components."""
        try:
            if self.model:
                self.model.save(self.model_path)
                print(f"💾 Model saved to {self.model_path}")
                
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
                
            print(f"💾 Preprocessing components saved")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained model and preprocessing components."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available for model loading")
            return False
            
        try:
            model_file = Path(model_path) if model_path else self.model_path
            
            if not model_file.exists():
                print(f"❌ Model file not found: {model_file}")
                return False
                
            # Load model
            self.model = load_model(model_file)
            
            # Load preprocessing components
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            print(f"✅ Model loaded successfully from {model_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
            
    def recognize_gesture(self, gesture_sequence: List[Dict]) -> Tuple[Optional[str], float]:
        """
        Recognize dynamic gesture from sequence.
        
        Args:
            gesture_sequence: List of frame data with detection_data
            
        Returns:
            Tuple of (predicted_letter, confidence)
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            print("❌ Model not available for recognition")
            return None, 0.0
            
        try:
            # Extract features
            feature_data = self.feature_extractor.extract_features(gesture_sequence)
            
            if not feature_data or feature_data['feature_vector'].size == 0:
                return None, 0.0
                
            feature_vector = feature_data['feature_vector']
            
            # Scale features
            feature_scaled = self.scaler.transform([feature_vector])
            
            # Predict
            prediction_probs = self.model.predict(feature_scaled, verbose=0)[0]
            
            # Get best prediction
            best_class_idx = np.argmax(prediction_probs)
            confidence = prediction_probs[best_class_idx]
            predicted_letter = self.label_encoder.inverse_transform([best_class_idx])[0]
            
            # Apply confidence threshold
            if confidence >= self.confidence_threshold:
                return predicted_letter, confidence
            else:
                return None, confidence
                
        except Exception as e:
            print(f"⚠️  Recognition error: {e}")
            return None, 0.0
            
    def recognize_with_history(self, gesture_sequence: List[Dict]) -> Tuple[Optional[str], float]:
        """Recognize gesture with temporal smoothing."""
        # Get current prediction
        prediction, confidence = self.recognize_gesture(gesture_sequence)
        
        # Update history
        self.recognition_history.append((prediction, confidence))
        
        # Keep history size reasonable
        if len(self.recognition_history) > self.history_size:
            self.recognition_history.pop(0)
            
        # Get stable prediction from history
        return self._get_stable_prediction()
        
    def _get_stable_prediction(self) -> Tuple[Optional[str], float]:
        """Get stable prediction from recognition history."""
        if len(self.recognition_history) == 0:
            return None, 0.0
            
        # Count predictions
        predictions = {}
        total_confidence = 0.0
        valid_count = 0
        
        for pred, conf in self.recognition_history:
            if pred is not None:
                predictions[pred] = predictions.get(pred, 0) + 1
                total_confidence += conf
                valid_count += 1
                
        if valid_count == 0:
            return None, 0.0
            
        # Get most common prediction
        most_common = max(predictions, key=predictions.get)
        avg_confidence = total_confidence / valid_count
        
        # Require consistency for stability
        consistency = predictions[most_common] / len(self.recognition_history)
        
        if consistency >= 0.6:  # 60% consistency required
            return most_common, avg_confidence
        else:
            return None, avg_confidence
            
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate model performance on test data."""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {}
            
        try:
            # Load data for evaluation
            if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
                if not self.load_training_data():
                    return {}
                    
            X, y = self.X_train, self.y_train
            
            # Encode and scale
            y_encoded = self.label_encoder.transform(y)
            X_scaled = self.scaler.transform(X)
            
            # Predictions
            y_pred_probs = self.model.predict(X_scaled, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(y_encoded == y_pred)
            
            # Per-class accuracy
            class_accuracies = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                mask = y_encoded == i
                if np.sum(mask) > 0:
                    class_acc = np.mean(y_pred[mask] == y_encoded[mask])
                    class_accuracies[class_name] = class_acc
                    
            return {
                'overall_accuracy': accuracy,
                'class_accuracies': class_accuracies,
                'total_samples': len(X),
                'classes': self.label_encoder.classes_.tolist()
            }
            
        except Exception as e:
            print(f"❌ Error evaluating model: {e}")
            return {}
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = {
            'model_available': self.model is not None,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'dynamic_letters': self.dynamic_letters,
            'confidence_threshold': self.confidence_threshold,
            'sequence_length': self.sequence_length
        }
        
        if self.model:
            info.update({
                'model_path': str(self.model_path),
                'total_parameters': self.model.count_params(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape
            })
            
        return info


def main():
    """Main function for testing dynamic recognition."""
    print("🧠 Dynamic Gesture Recognizer")
    print("=" * 40)
    
    recognizer = DynamicGestureRecognizer()
    
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Please install with:")
        print("   pip install tensorflow")
        return
        
    # Train model if no existing model
    if not recognizer.model:
        print("🔄 No existing model found. Training new model...")
        if recognizer.train_model():
            print("✅ Model training completed!")
        else:
            print("❌ Model training failed!")
            return
    else:
        print("✅ Using existing trained model")
        
    # Evaluate model
    evaluation = recognizer.evaluate_model()
    if evaluation:
        print(f"\n📊 Model Performance:")
        print(f"   • Overall accuracy: {evaluation['overall_accuracy']:.3f}")
        for letter, acc in evaluation['class_accuracies'].items():
            print(f"   • {letter} accuracy: {acc:.3f}")
            
    # Show model info
    info = recognizer.get_model_info()
    print(f"\n🔧 Model Info:")
    for key, value in info.items():
        print(f"   • {key}: {value}")


if __name__ == "__main__":
    main()