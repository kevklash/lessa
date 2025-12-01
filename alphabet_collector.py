"""
Dedicated Alphabet Collection Tool for LESSA
Systematic collection of ASL/LESSA alphabet letters with visual guides and progress tracking.
"""

import cv2
import sys
import os
import numpy as np
import time
import json
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.enhanced_camera import EnhancedCamera, CameraManager
from src.detection.holistic_detector import HolisticDetector
from src.detection.holistic_feature_extractor import HolisticFeatureExtractor

class AlphabetCollector:
    """Dedicated tool for collecting LESSA alphabet data."""
    
    def __init__(self):
        """Initialize alphabet collector."""
        self.alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.current_letter_index = 0
        self.target_samples_per_letter = 10
        self.collected_data = {}
        self.sample_count = 0
        
        # Initialize collected_data structure
        for letter in self.alphabet:
            self.collected_data[letter] = []
        
        # Load existing data
        self._load_existing_data()
        
        # Demo state
        self.show_guide = True
        self.show_progress = True
        self.show_help = True
        self.collection_mode = False
        
        print("🔤 LESSA Alphabet Collector Initialized!")
        print(f"📝 Target: {self.target_samples_per_letter} samples per letter")
    
    def _load_existing_data(self):
        """Load existing alphabet data if available."""
        try:
            data_file = 'lessa_alphabet_data.json'
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    saved_data = json.load(f)
                    
                # Count existing samples
                total_loaded = 0
                for letter in self.alphabet:
                    if letter in saved_data:
                        self.collected_data[letter] = saved_data[letter]
                        total_loaded += len(saved_data[letter])
                
                if total_loaded > 0:
                    print(f"📊 Loaded {total_loaded} existing samples from {data_file}")
        except Exception as e:
            print(f"⚠️  Could not load existing data: {e}")
    
    def get_current_letter(self) -> str:
        """Get currently selected letter."""
        return self.alphabet[self.current_letter_index]
    
    def get_letter_progress(self, letter: str) -> tuple:
        """Get progress for a specific letter."""
        collected = len(self.collected_data[letter])
        return collected, self.target_samples_per_letter
    
    def get_overall_progress(self) -> tuple:
        """Get overall collection progress."""
        total_collected = sum(len(samples) for samples in self.collected_data.values())
        total_target = len(self.alphabet) * self.target_samples_per_letter
        return total_collected, total_target
    
    def next_letter(self):
        """Move to next letter."""
        self.current_letter_index = (self.current_letter_index + 1) % len(self.alphabet)
        print(f"📝 Current letter: {self.get_current_letter()}")
    
    def previous_letter(self):
        """Move to previous letter."""
        self.current_letter_index = (self.current_letter_index - 1) % len(self.alphabet)
        print(f"📝 Current letter: {self.get_current_letter()}")
    
    def jump_to_next_incomplete(self):
        """Jump to next letter that needs more samples."""
        start_index = self.current_letter_index
        while True:
            self.next_letter()
            current_count, target = self.get_letter_progress(self.get_current_letter())
            if current_count < target or self.current_letter_index == start_index:
                break
        print(f"🎯 Jumped to letter: {self.get_current_letter()}")
    
    def save_sample(self, detection_data: dict, feature_extractor: HolisticFeatureExtractor):
        """Save a sample for the current letter."""
        try:
            current_letter = self.get_current_letter()
            
            # Extract features
            features = feature_extractor.extract_features(detection_data)
            
            # Create sample data
            sample_data = {
                'letter': current_letter,
                'sample_id': len(self.collected_data[current_letter]) + 1,
                'timestamp': datetime.now().isoformat(),
                'detection_data': self._serialize_detection_data(detection_data),
                'features_summary': {
                    'hands_detected': len([h for h in [detection_data['hands']['left_hand'], 
                                                      detection_data['hands']['right_hand']] if h]),
                    'pose_detected': detection_data['pose'] is not None,
                    'face_detected': detection_data['face'] is not None
                }
            }
            
            # Add to collection
            self.collected_data[current_letter].append(sample_data)
            
            # Save to file
            self._save_to_file()
            
            current_count, target = self.get_letter_progress(current_letter)
            print(f"✅ Saved '{current_letter}' sample {current_count}/{target}")
            
            # Auto-advance if letter is complete
            if current_count >= target:
                print(f"🎉 Letter '{current_letter}' complete! Moving to next incomplete letter.")
                self.jump_to_next_incomplete()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving sample: {e}")
            return False
    
    def _serialize_detection_data(self, detection_data: dict) -> dict:
        """Serialize detection data for JSON storage."""
        serialized = {
            'hands': {},
            'pose': None,
            'face': None,
            'detection_confidence': detection_data['detection_confidence']
        }
        
        # Serialize hands
        for hand_side in ['left_hand', 'right_hand']:
            hand_data = detection_data['hands'][hand_side]
            if hand_data:
                serialized['hands'][hand_side] = {
                    'landmarks': [[float(coord) for coord in landmark] 
                                 for landmark in hand_data['landmarks']],
                    'type': hand_data['type'],
                    'landmark_count': hand_data['landmark_count']
                }
            else:
                serialized['hands'][hand_side] = None
        
        # Serialize pose and face (simplified)
        for component in ['pose', 'face']:
            component_data = detection_data[component]
            if component_data:
                serialized[component] = {
                    'landmark_count': component_data['landmark_count'],
                    'type': component_data['type']
                }
        
        return serialized
    
    def _save_to_file(self):
        """Save all collected data to file."""
        try:
            filename = 'lessa_alphabet_data.json'
            with open(filename, 'w') as f:
                json.dump(self.collected_data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving to file: {e}")
    
    def draw_interface(self, frame: np.ndarray, detection_data: dict) -> np.ndarray:
        """Draw the alphabet collection interface."""
        interface_frame = frame.copy()
        
        # Draw header
        interface_frame = self._draw_header(interface_frame)
        
        # Draw letter guide
        if self.show_guide:
            interface_frame = self._draw_letter_guide(interface_frame)
        
        # Draw progress panel
        if self.show_progress:
            interface_frame = self._draw_progress_panel(interface_frame)
        
        # Draw help
        if self.show_help:
            interface_frame = self._draw_help_panel(interface_frame)
        
        # Draw collection status
        interface_frame = self._draw_collection_status(interface_frame, detection_data)
        
        return interface_frame
    
    def _draw_header(self, frame: np.ndarray) -> np.ndarray:
        """Draw header with current letter and status."""
        header_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Header background
        cv2.rectangle(header_frame, (0, 0), (width, 80), (0, 0, 0), -1)
        
        # Title
        cv2.putText(header_frame, "LESSA Alphabet Collector", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current letter (large)
        current_letter = self.get_current_letter()
        cv2.putText(header_frame, f"Letter: {current_letter}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Progress for current letter
        current_count, target = self.get_letter_progress(current_letter)
        progress_text = f"Progress: {current_count}/{target}"
        cv2.putText(header_frame, progress_text, 
                   (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return header_frame
    
    def _draw_letter_guide(self, frame: np.ndarray) -> np.ndarray:
        """Draw visual guide for current letter."""
        guide_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Guide panel background
        panel_width = 300
        panel_height = 200
        start_x = width - panel_width - 10
        start_y = 90
        
        overlay = guide_frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + panel_width, start_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, guide_frame, 0.2, 0, guide_frame)
        
        # Title
        cv2.putText(guide_frame, "Letter Guide", 
                   (start_x + 10, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Large letter display
        current_letter = self.get_current_letter()
        cv2.putText(guide_frame, current_letter, 
                   (start_x + 130, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4)
        
        # Instructions
        instructions = [
            "1. Form the letter with your hand",
            "2. Hold steady when detected",
            "3. Press SPACE to save sample",
            "4. Use N/P for next/previous"
        ]
        
        y_offset = start_y + 110
        for instruction in instructions:
            cv2.putText(guide_frame, instruction, 
                       (start_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
        
        return guide_frame
    
    def _draw_progress_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw alphabet progress overview."""
        progress_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Panel setup
        panel_width = 400
        panel_height = 150
        start_x = 10
        start_y = height - panel_height - 10
        
        # Background
        overlay = progress_frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + panel_width, start_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, progress_frame, 0.2, 0, progress_frame)
        
        # Title
        cv2.putText(progress_frame, "Alphabet Progress", 
                   (start_x + 10, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Overall progress
        total_collected, total_target = self.get_overall_progress()
        overall_text = f"Overall: {total_collected}/{total_target} samples"
        cv2.putText(progress_frame, overall_text, 
                   (start_x + 10, start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alphabet grid (5x6 to show all 26 letters)
        grid_start_x = start_x + 10
        grid_start_y = start_y + 70
        cell_size = 15
        
        for i, letter in enumerate(self.alphabet):
            row = i // 13  # 13 letters per row
            col = i % 13
            
            x = grid_start_x + col * (cell_size + 2)
            y = grid_start_y + row * (cell_size + 5)
            
            # Color based on completion
            collected, target = self.get_letter_progress(letter)
            if collected >= target:
                color = (0, 255, 0)  # Green - complete
            elif collected > 0:
                color = (0, 255, 255)  # Yellow - in progress
            else:
                color = (128, 128, 128)  # Gray - not started
            
            # Highlight current letter
            if i == self.current_letter_index:
                cv2.rectangle(progress_frame, (x-2, y-12), (x+12, y+2), (255, 255, 255), 2)
            
            # Draw letter
            cv2.putText(progress_frame, letter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return progress_frame
    
    def _draw_help_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw help panel with controls."""
        help_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Panel setup
        panel_width = 280
        panel_height = 160
        start_x = width - panel_width - 10
        start_y = height - panel_height - 10
        
        # Background
        overlay = help_frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + panel_width, start_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, help_frame, 0.2, 0, help_frame)
        
        # Controls
        controls = [
            "Alphabet Controls:",
            "SPACE - Save sample",
            "N - Next letter", 
            "P - Previous letter",
            "J - Jump to incomplete",
            "G - Toggle guide",
            "R - Toggle progress",
            "H - Toggle help",
            "Q - Quit"
        ]
        
        y_offset = start_y + 20
        for i, control in enumerate(controls):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            font_size = 0.5 if i == 0 else 0.4
            cv2.putText(help_frame, control, 
                       (start_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
            y_offset += 16
        
        return help_frame
    
    def _draw_collection_status(self, frame: np.ndarray, detection_data: dict) -> np.ndarray:
        """Draw collection status overlay."""
        status_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Detection status
        hands_detected = len([h for h in [detection_data['hands']['left_hand'], 
                                         detection_data['hands']['right_hand']] if h])
        
        if hands_detected > 0:
            status_text = f"✅ {hands_detected} hand(s) detected - Press SPACE to save"
            color = (0, 255, 0)
        else:
            status_text = "❌ No hands detected - Position your hand to form the letter"
            color = (0, 0, 255)
        
        # Status background
        cv2.rectangle(status_frame, (10, height - 40), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.putText(status_frame, status_text, 
                   (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return status_frame

def run_alphabet_collector():
    """Run the alphabet collection tool."""
    print("🔤 Starting LESSA Alphabet Collector...")
    print("=" * 50)
    
    # Detect cameras
    available_cameras = CameraManager.detect_cameras()
    if not available_cameras:
        print("❌ No cameras detected.")
        return
    
    # Select camera
    best_camera = CameraManager.get_best_camera()
    print(f"📹 Using Camera {best_camera.device_id}")
    
    # Initialize components
    camera = EnhancedCamera(best_camera.device_id)
    holistic_detector = HolisticDetector()
    feature_extractor = HolisticFeatureExtractor()
    alphabet_collector = AlphabetCollector()
    
    try:
        # Start camera
        if not camera.start(optimize_quality=True):
            print("❌ Failed to start camera")
            return
        
        print(f"📝 Current letter: {alphabet_collector.get_current_letter()}")
        print("🎯 Controls: SPACE=Save | N/P=Next/Prev | J=Jump | G/R/H=Toggle | Q=Quit")
        
        while True:
            # Read frame
            ret, frame = camera.read_frame()
            if not ret:
                break
            
            # Detect holistic data
            annotated_frame, detection_data = holistic_detector.detect_holistic(frame)
            
            # Draw alphabet interface
            interface_frame = alphabet_collector.draw_interface(annotated_frame, detection_data)
            
            # Display frame
            cv2.imshow('LESSA Alphabet Collector', interface_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to save
                if holistic_detector.is_detection_complete(detection_data):
                    alphabet_collector.save_sample(detection_data, feature_extractor)
                else:
                    print("⚠️  No hands detected - cannot save sample")
            elif key == ord('n'):
                alphabet_collector.next_letter()
            elif key == ord('p'):
                alphabet_collector.previous_letter()
            elif key == ord('j'):
                alphabet_collector.jump_to_next_incomplete()
            elif key == ord('g'):
                alphabet_collector.show_guide = not alphabet_collector.show_guide
                print(f"📋 Guide display: {'ON' if alphabet_collector.show_guide else 'OFF'}")
            elif key == ord('r'):
                alphabet_collector.show_progress = not alphabet_collector.show_progress
                print(f"📊 Progress display: {'ON' if alphabet_collector.show_progress else 'OFF'}")
            elif key == ord('h'):
                alphabet_collector.show_help = not alphabet_collector.show_help
                print(f"💡 Help display: {'ON' if alphabet_collector.show_help else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Collection interrupted by user")
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        camera.stop()
        holistic_detector.close()
        cv2.destroyAllWindows()
        
        # Show final statistics
        total_collected, total_target = alphabet_collector.get_overall_progress()
        print(f"\n📊 Final Statistics:")
        print(f"   Total samples: {total_collected}/{total_target}")
        
        completed_letters = [letter for letter in alphabet_collector.alphabet 
                           if len(alphabet_collector.collected_data[letter]) >= alphabet_collector.target_samples_per_letter]
        print(f"   Completed letters: {len(completed_letters)}/26")
        if completed_letters:
            print(f"   Letters completed: {', '.join(completed_letters)}")
        
        print("🔤 Alphabet collection ended.")

if __name__ == "__main__":
    run_alphabet_collector()