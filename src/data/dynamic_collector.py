"""
Dynamic gesture data collector for movement-based LESSA letters (J, Z).
Based on research findings: LSTM + MediaPipe + motion-based segmentation approach.
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.detection.holistic_detector import HolisticDetector
from src.utils.enhanced_camera import EnhancedCamera


class DynamicGestureCollector:
    """Collect temporal sequences for dynamic LESSA letters."""
    
    def __init__(self):
        """Initialize dynamic gesture collector."""
        self.camera = EnhancedCamera()
        self.detector = HolisticDetector()
        
        # Dynamic letters that require movement
        self.dynamic_letters = ['J', 'Z']
        
        # Sequence parameters based on research
        self.sequence_length = 30  # frames (1 second at 30fps)
        self.motion_threshold = 0.05  # Movement detection threshold
        self.min_gesture_frames = 15  # Minimum frames for valid gesture
        self.max_idle_frames = 90  # Max frames to wait before timeout (3 seconds)
        
        # Collection state
        self.current_letter = None
        self.collecting = False
        self.gesture_sequence = []
        self.motion_history = []
        self.idle_frames = 0
        self.sample_count = 0
        
        # Data storage
        self.data_file = Path("lessa_dynamic_data.json")
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing dynamic gesture data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    self.data = json.load(f)
                print(f"📁 Loaded existing dynamic data: {len(self.data)} letters")
            except Exception as e:
                print(f"⚠️  Error loading data: {e}")
                self.data = {}
        else:
            self.data = {}
            
        # Count existing samples
        for letter in self.dynamic_letters:
            if letter in self.data:
                self.sample_count = len(self.data[letter])
                
    def save_data(self):
        """Save collected data to file."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"💾 Data saved successfully")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            
    def calculate_motion(self, current_landmarks: np.ndarray, 
                        previous_landmarks: np.ndarray) -> float:
        """Calculate motion magnitude between frames."""
        try:
            if current_landmarks is None or previous_landmarks is None:
                return 0.0
                
            # Focus on hand landmarks for motion detection
            current_hands = self._extract_hand_landmarks(current_landmarks)
            previous_hands = self._extract_hand_landmarks(previous_landmarks)
            
            if current_hands is None or previous_hands is None:
                return 0.0
                
            # Calculate Euclidean distance between corresponding points
            motion = np.mean(np.linalg.norm(current_hands - previous_hands, axis=1))
            return motion
            
        except Exception as e:
            print(f"⚠️  Motion calculation error: {e}")
            return 0.0
            
    def _extract_hand_landmarks(self, detection_data: Dict) -> Optional[np.ndarray]:
        """Extract hand landmarks for motion calculation."""
        try:
            hands = detection_data.get('hands', {})
            
            # Prioritize right hand, fall back to left
            hand_data = hands.get('right_hand') or hands.get('left_hand')
            
            if hand_data and hand_data.get('landmarks'):
                landmarks = np.array(hand_data['landmarks'])[:, :3]  # x, y, z only
                return landmarks
                
            return None
            
        except Exception as e:
            return None
            
    def detect_gesture_boundaries(self) -> Tuple[bool, bool]:
        """
        Detect start and end of gesture based on motion analysis.
        Returns: (gesture_started, gesture_ended)
        """
        if len(self.motion_history) < 2:
            return False, False
            
        current_motion = self.motion_history[-1]
        
        # Gesture start: motion exceeds threshold
        gesture_started = (not self.collecting and 
                          current_motion > self.motion_threshold)
        
        # Gesture end: motion drops below threshold for enough frames
        gesture_ended = False
        if self.collecting and current_motion < self.motion_threshold:
            self.idle_frames += 1
            if self.idle_frames >= 10:  # 1/3 second of low motion
                gesture_ended = True
        else:
            self.idle_frames = 0
            
        return gesture_started, gesture_ended
        
    def collect_sequence(self, letter: str) -> bool:
        """
        Collect a dynamic gesture sequence for the specified letter.
        Returns True if sequence was successfully collected.
        """
        if letter not in self.dynamic_letters:
            print(f"❌ {letter} is not a dynamic letter")
            return False
            
        self.current_letter = letter
        self.reset_collection_state()
        
        print(f"\n🎯 Collecting dynamic gesture for letter '{letter}'")
        print("📋 Instructions:")
        print(f"   • Position your hand in view")
        print(f"   • Wait for 'READY' signal")
        print(f"   • Perform the {letter} gesture smoothly")
        print(f"   • Hold still when complete")
        print("   • Press 'q' to cancel, 'r' to retry")
        
        try:
            self.camera.start()
            
            while True:
                success, frame = self.camera.read_frame()
                if not success or frame is None:
                    continue
                    
                # Detect landmarks
                frame, detection_data = self.detector.detect_holistic(frame)
                
                # Calculate motion if we have previous frame
                current_motion = 0.0
                if len(self.gesture_sequence) > 0:
                    previous_detection = self.gesture_sequence[-1]['detection_data']
                    current_motion = self.calculate_motion(detection_data, previous_detection)
                    
                self.motion_history.append(current_motion)
                
                # Keep motion history reasonable size
                if len(self.motion_history) > 100:
                    self.motion_history.pop(0)
                    
                # Gesture boundary detection
                gesture_started, gesture_ended = self.detect_gesture_boundaries()
                
                if gesture_started:
                    print("🟢 RECORDING - Perform gesture now!")
                    self.collecting = True
                    self.gesture_sequence = []
                    
                if self.collecting:
                    # Add frame to sequence
                    frame_data = {
                        'detection_data': detection_data,
                        'timestamp': time.time(),
                        'motion': current_motion,
                        'frame_index': len(self.gesture_sequence)
                    }
                    self.gesture_sequence.append(frame_data)
                    
                    # Check if sequence is complete
                    if gesture_ended and len(self.gesture_sequence) >= self.min_gesture_frames:
                        if self._validate_and_save_sequence():
                            return True
                        else:
                            self.reset_collection_state()
                            print("🔄 Sequence invalid, try again...")
                            
                    elif len(self.gesture_sequence) >= self.sequence_length * 2:
                        print("⚠️  Sequence too long, please hold still to complete")
                        
                # Display frame with feedback
                self._draw_collection_feedback(frame, current_motion, detection_data)
                
                cv2.imshow('Dynamic Gesture Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("❌ Collection cancelled")
                    return False
                elif key == ord('r'):
                    print("🔄 Retrying...")
                    self.reset_collection_state()
                    
        except KeyboardInterrupt:
            print("\n❌ Collection interrupted")
            return False
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            
        return False
        
    def reset_collection_state(self):
        """Reset collection state for new attempt."""
        self.collecting = False
        self.gesture_sequence = []
        self.motion_history = []
        self.idle_frames = 0
        
    def _validate_and_save_sequence(self) -> bool:
        """Validate collected sequence and save if valid."""
        try:
            if len(self.gesture_sequence) < self.min_gesture_frames:
                print(f"❌ Sequence too short: {len(self.gesture_sequence)} frames")
                return False
                
            # Check for hand landmarks in most frames
            valid_frames = 0
            for frame_data in self.gesture_sequence:
                if self._extract_hand_landmarks(frame_data['detection_data']) is not None:
                    valid_frames += 1
                    
            if valid_frames < len(self.gesture_sequence) * 0.8:
                print(f"❌ Too many frames without hand detection: {valid_frames}/{len(self.gesture_sequence)}")
                return False
                
            # Create sample data
            sample_data = {
                'letter': self.current_letter,
                'timestamp': datetime.now().isoformat(),
                'sequence_length': len(self.gesture_sequence),
                'motion_profile': [frame['motion'] for frame in self.gesture_sequence],
                'landmarks_sequence': []
            }
            
            # Extract landmarks for each frame
            for frame_data in self.gesture_sequence:
                sample_data['landmarks_sequence'].append({
                    'detection_data': frame_data['detection_data'],
                    'timestamp': frame_data['timestamp'],
                    'motion': frame_data['motion']
                })
                
            # Save to data structure
            if self.current_letter not in self.data:
                self.data[self.current_letter] = []
                
            self.data[self.current_letter].append(sample_data)
            self.sample_count += 1
            
            print(f"✅ Sequence saved! ({len(self.gesture_sequence)} frames)")
            print(f"📊 Total samples for {self.current_letter}: {len(self.data[self.current_letter])}")
            
            self.save_data()
            return True
            
        except Exception as e:
            print(f"❌ Error validating sequence: {e}")
            return False
            
    def _draw_collection_feedback(self, frame: np.ndarray, motion: float, 
                                 detection_data: Dict):
        """Draw collection feedback on frame."""
        try:
            h, w = frame.shape[:2]
            
            # Status indicator
            if self.collecting:
                status_color = (0, 255, 0)  # Green
                status_text = "RECORDING"
            else:
                status_color = (0, 255, 255)  # Yellow
                status_text = "READY"
                
            cv2.rectangle(frame, (10, 10), (200, 50), status_color, -1)
            cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 0), 2)
                       
            # Motion indicator
            motion_text = f"Motion: {motion:.3f}"
            cv2.putText(frame, motion_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
                       
            # Sequence progress
            if self.collecting:
                progress_text = f"Frames: {len(self.gesture_sequence)}/{self.sequence_length}"
                cv2.putText(frame, progress_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                           
            # Letter being collected
            letter_text = f"Letter: {self.current_letter}"
            cv2.putText(frame, letter_text, (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 0), 2)
                       
            # Draw hand landmarks if available
            hands = detection_data.get('hands', {})
            for hand_name, hand_data in hands.items():
                if hand_data and hand_data.get('landmarks'):
                    self._draw_hand_landmarks(frame, hand_data['landmarks'])
                    
        except Exception as e:
            print(f"⚠️  Drawing error: {e}")
            
    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks: List):
        """Draw hand landmarks on frame."""
        try:
            h, w = frame.shape[:2]
            
            for landmark in landmarks:
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
        except Exception as e:
            pass
            
    def collect_samples_interactive(self) -> bool:
        """Interactive collection session for all dynamic letters."""
        print("\n🎬 Dynamic Gesture Collection Session")
        print("=" * 50)
        
        for letter in self.dynamic_letters:
            while True:
                print(f"\n📝 Ready to collect samples for letter '{letter}'")
                print(f"Current samples: {len(self.data.get(letter, []))}")
                
                choice = input("Collect sample? (y/n/q): ").lower().strip()
                
                if choice == 'q':
                    print("👋 Collection session ended")
                    return True
                elif choice == 'n':
                    break
                elif choice == 'y':
                    success = self.collect_sequence(letter)
                    if success:
                        print("✅ Sample collected successfully!")
                    else:
                        print("❌ Sample collection failed")
                else:
                    print("Please enter 'y', 'n', or 'q'")
                    
        print("\n🎉 All dynamic letters processed!")
        return True


def main():
    """Main function for testing dynamic collection."""
    collector = DynamicGestureCollector()
    collector.collect_samples_interactive()


if __name__ == "__main__":
    main()