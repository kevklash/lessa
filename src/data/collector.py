"""
Data collection tool for gathering sign language gesture samples.
"""

import cv2
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.camera import Camera
from src.utils.config import config
from src.utils.helpers import create_directory_if_not_exists, get_timestamp
from src.detection.hand_detector import HandDetector
from src.detection.feature_extractor import FeatureExtractor

class DataCollector:
    """Data collection tool for sign language gestures."""
    
    def __init__(self):
        """Initialize data collector."""
        self.camera = Camera()
        self.hand_detector = HandDetector()
        self.feature_extractor = FeatureExtractor()
        
        # Get configuration
        self.gestures = config.get_gestures().get('static', [])
        paths = config.get_paths()
        self.data_dir = paths.get('raw_data_dir', 'data/raw')
        
        # Create data directory
        create_directory_if_not_exists(self.data_dir)
        
        # Collection state
        self.current_gesture = None
        self.current_gesture_index = 0
        self.samples_collected = 0
        self.target_samples = config.get('data_collection', {}).get('samples_per_gesture', 100)
        self.recording = False
        self.countdown = 0
        
        print(f"Data Collector initialized")
        print(f"Gestures to collect: {len(self.gestures)}")
        print(f"Samples per gesture: {self.target_samples}")
        print(f"Data directory: {self.data_dir}")
    
    def start_collection(self):
        """Start the data collection process."""
        if not self.camera.start():
            print("Failed to start camera")
            return
        
        if not self.gestures:
            print("No gestures configured for collection")
            return
        
        print("\n=== DATA COLLECTION STARTED ===")
        print("Controls:")
        print("SPACE - Start/Stop recording current gesture")
        print("N - Next gesture")
        print("P - Previous gesture")
        print("R - Reset current gesture samples")
        print("Q - Quit")
        print("=" * 35)
        
        self.current_gesture = self.gestures[0]
        
        try:
            while True:
                ret, frame = self.camera.read_frame()
                if not ret:
                    break
                
                # Detect hands
                annotated_frame, hands_data = self.hand_detector.detect_hands(frame)
                
                # Add collection interface
                display_frame = self._draw_collection_interface(annotated_frame, hands_data)
                
                # Handle recording
                if self.recording and hands_data:
                    self._record_sample(hands_data)
                
                cv2.imshow('Sign Language Data Collection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space bar
                    self._toggle_recording()
                elif key == ord('n'):
                    self._next_gesture()
                elif key == ord('p'):
                    self._previous_gesture()
                elif key == ord('r'):
                    self._reset_current_gesture()
        
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        finally:
            self._cleanup()
    
    def _draw_collection_interface(self, frame, hands_data):
        """Draw the data collection interface on the frame."""
        display_frame = frame.copy()
        
        # Background for text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Current gesture info
        cv2.putText(display_frame, f"Gesture: {self.current_gesture} ({self.current_gesture_index + 1}/{len(self.gestures)})", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Samples: {self.samples_collected}/{self.target_samples}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Recording status
        status_color = (0, 0, 255) if self.recording else (255, 255, 255)
        status_text = "RECORDING" if self.recording else "READY"
        cv2.putText(display_frame, f"Status: {status_text}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Hand detection status
        hands_status = f"Hands detected: {len(hands_data)}"
        hands_color = (0, 255, 0) if hands_data else (0, 0, 255)
        cv2.putText(display_frame, hands_status, 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hands_color, 2)
        
        # Progress bar
        progress_width = 400
        progress_height = 20
        progress_x = 20
        progress_y = 160
        
        # Background
        cv2.rectangle(display_frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + progress_height), (50, 50, 50), -1)
        
        # Progress
        if self.target_samples > 0:
            progress = min(self.samples_collected / self.target_samples, 1.0)
            progress_fill = int(progress_width * progress)
            cv2.rectangle(display_frame, (progress_x, progress_y), 
                         (progress_x + progress_fill, progress_y + progress_height), (0, 255, 0), -1)
        
        # Progress text
        cv2.putText(display_frame, f"{self.samples_collected}/{self.target_samples}", 
                   (progress_x + progress_width//2 - 50, progress_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def _toggle_recording(self):
        """Toggle recording state."""
        self.recording = not self.recording
        if self.recording:
            print(f"Started recording gesture: {self.current_gesture}")
        else:
            print(f"Stopped recording gesture: {self.current_gesture}")
    
    def _record_sample(self, hands_data):
        """Record a sample for the current gesture."""
        if not hands_data:
            return
        
        # Use the first detected hand for now
        hand_data = hands_data[0]
        
        # Extract features
        features = self.feature_extractor.extract_features(hand_data)
        
        # Create sample data
        sample_data = {
            'gesture': self.current_gesture,
            'timestamp': datetime.now().isoformat(),
            'hand_data': hand_data,
            'features': {
                'raw_landmarks': features['raw_landmarks'].tolist(),
                'normalized_landmarks': features['normalized_landmarks'].tolist(),
                'distances': features['distances'],
                'angles': features['angles'],
                'hand_orientation': features['hand_orientation']
            }
        }
        
        # Save sample
        self._save_sample(sample_data)
        self.samples_collected += 1
        
        # Check if we've collected enough samples for this gesture
        if self.samples_collected >= self.target_samples:
            print(f"Completed collection for gesture: {self.current_gesture}")
            self.recording = False
            
            # Auto advance to next gesture if available
            if self.current_gesture_index < len(self.gestures) - 1:
                self._next_gesture()
            else:
                print("All gestures completed!")
    
    def _save_sample(self, sample_data):
        """Save a sample to disk."""
        gesture_dir = os.path.join(self.data_dir, self.current_gesture)
        create_directory_if_not_exists(gesture_dir)
        
        filename = f"{self.current_gesture}_{get_timestamp()}_{self.samples_collected:04d}.json"
        filepath = os.path.join(gesture_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(sample_data, f, indent=2)
        except Exception as e:
            print(f"Error saving sample: {e}")
    
    def _next_gesture(self):
        """Move to next gesture."""
        if self.current_gesture_index < len(self.gestures) - 1:
            self.current_gesture_index += 1
            self.current_gesture = self.gestures[self.current_gesture_index]
            self._load_existing_samples()
            print(f"Switched to gesture: {self.current_gesture}")
        else:
            print("Already at last gesture")
    
    def _previous_gesture(self):
        """Move to previous gesture."""
        if self.current_gesture_index > 0:
            self.current_gesture_index -= 1
            self.current_gesture = self.gestures[self.current_gesture_index]
            self._load_existing_samples()
            print(f"Switched to gesture: {self.current_gesture}")
        else:
            print("Already at first gesture")
    
    def _reset_current_gesture(self):
        """Reset samples for current gesture."""
        self.samples_collected = 0
        print(f"Reset samples for gesture: {self.current_gesture}")
    
    def _load_existing_samples(self):
        """Load count of existing samples for current gesture."""
        gesture_dir = os.path.join(self.data_dir, self.current_gesture)
        if os.path.exists(gesture_dir):
            existing_files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
            self.samples_collected = len(existing_files)
        else:
            self.samples_collected = 0
    
    def _cleanup(self):
        """Clean up resources."""
        self.camera.stop()
        self.hand_detector.close()
        cv2.destroyAllWindows()
        print("Data collection ended. Resources cleaned up.")

def main():
    """Main function to run data collection."""
    collector = DataCollector()
    collector.start_collection()

if __name__ == "__main__":
    main()