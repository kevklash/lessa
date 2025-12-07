"""
Enhanced LESSA demo with both static and dynamic gesture recognition.
Integrates alphabet recognition for static letters and LSTM-based recognition for dynamic letters (J, Z).
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from src.detection.holistic_detector import HolisticDetector
from src.utils.enhanced_camera import EnhancedCamera
from alphabet_recognizer import AlphabetRecognizer
from dynamic_gesture_recognizer import DynamicGestureRecognizer
from src.data.dynamic_collector import DynamicGestureCollector


class EnhancedLESSADemo:
    """Enhanced LESSA demo with static and dynamic gesture recognition."""
    
    def __init__(self):
        """Initialize enhanced LESSA demo."""
        self.camera = EnhancedCamera()
        self.detector = HolisticDetector()
        
        # Recognition systems
        self.static_recognizer = AlphabetRecognizer()
        self.dynamic_recognizer = DynamicGestureRecognizer()
        
        # Letter categories
        self.static_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) 
                              if chr(i) not in ['J', 'Z']]
        self.dynamic_letters = ['J', 'Z']
        
        # Recognition state
        self.current_mode = 'static'  # 'static' or 'dynamic'
        self.dynamic_sequence = []
        self.dynamic_collecting = False
        self.dynamic_start_time = None
        self.motion_history = []
        self.motion_threshold = 0.05
        
        # Display parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # User interface state
        self.show_landmarks = True
        self.show_confidence = True
        self.show_help = False
        
    def initialize_systems(self) -> bool:
        """Initialize recognition systems."""
        print("🚀 Initializing Enhanced LESSA Recognition Systems...")
        
        # Initialize static recognizer
        print("📊 Loading static alphabet recognizer...")
        if not self.static_recognizer.load_training_data():
            print("❌ Failed to load static training data")
            return False
        print("✅ Static recognizer ready")
        
        # Check dynamic recognizer
        print("🧠 Checking dynamic gesture recognizer...")
        dynamic_info = self.dynamic_recognizer.get_model_info()
        
        if dynamic_info['model_available']:
            print("✅ Dynamic recognizer ready")
        else:
            print("⚠️  No dynamic model found")
            print("   Run dynamic training first or collect dynamic samples")
            
        return True
        
    def detect_mode_switch(self, detection_data: Dict) -> str:
        """
        Detect whether to use static or dynamic recognition based on motion.
        """
        try:
            # Calculate current motion
            current_motion = self._calculate_motion(detection_data)
            self.motion_history.append(current_motion)
            
            # Keep motion history reasonable
            if len(self.motion_history) > 30:  # 1 second at 30fps
                self.motion_history.pop(0)
                
            # Calculate average motion over recent frames
            if len(self.motion_history) >= 10:
                avg_motion = np.mean(self.motion_history[-10:])
                
                # Switch to dynamic mode if significant motion detected
                if avg_motion > self.motion_threshold and not self.dynamic_collecting:
                    return 'dynamic'
                # Switch to static mode if motion is low
                elif avg_motion < self.motion_threshold * 0.5 and self.current_mode == 'dynamic':
                    return 'static'
                    
            return self.current_mode
            
        except Exception as e:
            print(f"⚠️  Mode detection error: {e}")
            return 'static'
            
    def _calculate_motion(self, current_detection: Dict) -> float:
        """Calculate motion magnitude for mode detection."""
        try:
            if not hasattr(self, 'previous_detection') or self.previous_detection is None:
                self.previous_detection = current_detection
                return 0.0
                
            current_hands = self._extract_hand_landmarks(current_detection)
            previous_hands = self._extract_hand_landmarks(self.previous_detection)
            
            if current_hands is None or previous_hands is None:
                self.previous_detection = current_detection
                return 0.0
                
            # Calculate average landmark displacement
            motion = np.mean(np.linalg.norm(current_hands - previous_hands, axis=1))
            
            self.previous_detection = current_detection
            return motion
            
        except Exception as e:
            return 0.0
            
    def _extract_hand_landmarks(self, detection_data: Dict) -> Optional[np.ndarray]:
        """Extract hand landmarks for motion calculation."""
        try:
            hands = detection_data.get('hands', {})
            hand_data = hands.get('right_hand') or hands.get('left_hand')
            
            if hand_data and hand_data.get('landmarks'):
                landmarks = np.array(hand_data['landmarks'])[:, :3]
                return landmarks
                
            return None
            
        except Exception:
            return None
            
    def process_dynamic_recognition(self, detection_data: Dict) -> Tuple[Optional[str], float]:
        """Process dynamic gesture recognition."""
        try:
            current_motion = self.motion_history[-1] if self.motion_history else 0.0
            
            # Start collecting if motion begins
            if not self.dynamic_collecting and current_motion > self.motion_threshold:
                self.dynamic_collecting = True
                self.dynamic_sequence = []
                self.dynamic_start_time = time.time()
                return None, 0.0
                
            # Collect frames during motion
            if self.dynamic_collecting:
                # Add frame to sequence
                frame_data = {
                    'detection_data': detection_data,
                    'timestamp': time.time(),
                    'motion': current_motion
                }
                self.dynamic_sequence.append(frame_data)
                
                # Check for end of gesture (low motion for several frames)
                if len(self.dynamic_sequence) >= 10:  # Minimum frames
                    recent_motion = np.mean([f['motion'] for f in self.dynamic_sequence[-5:]])
                    
                    if (recent_motion < self.motion_threshold * 0.3 or 
                        len(self.dynamic_sequence) > 60):  # End conditions
                        
                        # Process collected sequence
                        if len(self.dynamic_sequence) >= 15:  # Minimum for valid gesture
                            prediction, confidence = self.dynamic_recognizer.recognize_gesture(
                                self.dynamic_sequence
                            )
                            
                            # Reset collection
                            self.dynamic_collecting = False
                            self.dynamic_sequence = []
                            
                            return prediction, confidence
                        else:
                            # Sequence too short, reset
                            self.dynamic_collecting = False
                            self.dynamic_sequence = []
                            
            return None, 0.0
            
        except Exception as e:
            print(f"⚠️  Dynamic recognition error: {e}")
            return None, 0.0
            
    def run_demo(self):
        """Run the enhanced LESSA demo."""
        print("\n🎬 Starting Enhanced LESSA Demo")
        print("=" * 50)
        print("Controls:")
        print("  • 's' - Switch to static mode")
        print("  • 'd' - Switch to dynamic mode") 
        print("  • 'a' - Auto mode (motion-based switching)")
        print("  • 'l' - Toggle landmark display")
        print("  • 'c' - Toggle confidence display")
        print("  • 'h' - Toggle help")
        print("  • 'q' - Quit")
        print("=" * 50)
        
        try:
            self.camera.start()
            
            # Recognition state
            current_prediction = None
            current_confidence = 0.0
            last_recognition_time = 0
            recognition_cooldown = 0.5  # seconds
            auto_mode = True
            
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                    
                # Update FPS
                self._update_fps()
                
                # Detect landmarks
                detection_data = self.detector.detect(frame)
                
                # Mode switching logic
                if auto_mode:
                    new_mode = self.detect_mode_switch(detection_data)
                    if new_mode != self.current_mode:
                        self.current_mode = new_mode
                        print(f"🔄 Switched to {self.current_mode} mode")
                        
                # Recognition logic
                if time.time() - last_recognition_time > recognition_cooldown:
                    if self.current_mode == 'static':
                        # Static recognition
                        prediction, confidence = self.static_recognizer.recognize_alphabet(detection_data)
                        if prediction:
                            current_prediction = prediction
                            current_confidence = confidence
                            last_recognition_time = time.time()
                            
                    elif self.current_mode == 'dynamic':
                        # Dynamic recognition
                        prediction, confidence = self.process_dynamic_recognition(detection_data)
                        if prediction:
                            current_prediction = prediction
                            current_confidence = confidence
                            last_recognition_time = time.time()
                            
                # Draw interface
                self._draw_interface(frame, detection_data, current_prediction, 
                                   current_confidence, auto_mode)
                
                cv2.imshow('Enhanced LESSA Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Demo ended")
                    break
                elif key == ord('s'):
                    self.current_mode = 'static'
                    auto_mode = False
                    print("🔧 Manual static mode")
                elif key == ord('d'):
                    self.current_mode = 'dynamic'
                    auto_mode = False
                    print("🔧 Manual dynamic mode")
                elif key == ord('a'):
                    auto_mode = True
                    print("🤖 Auto mode enabled")
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                elif key == ord('h'):
                    self.show_help = not self.show_help
                    
        except KeyboardInterrupt:
            print("\n❌ Demo interrupted")
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
            
    def _draw_interface(self, frame: np.ndarray, detection_data: Dict, 
                       prediction: Optional[str], confidence: float, auto_mode: bool):
        """Draw the user interface on the frame."""
        try:
            h, w = frame.shape[:2]
            
            # Draw landmarks if enabled
            if self.show_landmarks:
                self._draw_landmarks(frame, detection_data)
                
            # Status panel background
            panel_height = 120
            cv2.rectangle(frame, (0, 0), (300, panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (300, panel_height), (255, 255, 255), 2)
            
            # Mode indicator
            mode_color = (0, 255, 0) if self.current_mode == 'static' else (255, 100, 0)
            mode_text = f"Mode: {self.current_mode.upper()}"
            cv2.putText(frame, mode_text, (10, 25), self.font, 0.7, mode_color, 2)
            
            # Auto mode indicator
            auto_text = "AUTO" if auto_mode else "MANUAL"
            auto_color = (0, 255, 255) if auto_mode else (128, 128, 128)
            cv2.putText(frame, auto_text, (200, 25), self.font, 0.6, auto_color, 2)
            
            # Current prediction
            if prediction:
                pred_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                pred_text = f"Letter: {prediction}"
                cv2.putText(frame, pred_text, (10, 55), self.font, 0.8, pred_color, 2)
                
                if self.show_confidence:
                    conf_text = f"Conf: {confidence:.2f}"
                    cv2.putText(frame, conf_text, (10, 80), self.font, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Letter: ---", (10, 55), self.font, 0.8, (128, 128, 128), 2)
                
            # Dynamic collection indicator
            if self.dynamic_collecting:
                collect_text = f"Recording... ({len(self.dynamic_sequence)})"
                cv2.putText(frame, collect_text, (10, 105), self.font, 0.6, (0, 0, 255), 2)
                
            # FPS counter
            fps_text = f"FPS: {self.current_fps}"
            cv2.putText(frame, fps_text, (w - 100, 25), self.font, 0.6, (255, 255, 255), 2)
            
            # Motion indicator
            if self.motion_history:
                current_motion = self.motion_history[-1]
                motion_text = f"Motion: {current_motion:.3f}"
                motion_color = (0, 255, 0) if current_motion > self.motion_threshold else (128, 128, 128)
                cv2.putText(frame, motion_text, (w - 200, 50), self.font, 0.5, motion_color, 1)
                
            # Help panel
            if self.show_help:
                self._draw_help_panel(frame)
                
        except Exception as e:
            print(f"⚠️  Interface drawing error: {e}")
            
    def _draw_landmarks(self, frame: np.ndarray, detection_data: Dict):
        """Draw hand landmarks on the frame."""
        try:
            hands = detection_data.get('hands', {})
            h, w = frame.shape[:2]
            
            for hand_name, hand_data in hands.items():
                if hand_data and hand_data.get('landmarks'):
                    landmarks = hand_data['landmarks']
                    
                    # Draw landmarks
                    for i, landmark in enumerate(landmarks):
                        x = int(landmark[0] * w)
                        y = int(landmark[1] * h)
                        
                        # Different colors for different landmarks
                        if i in [4, 8, 12, 16, 20]:  # Fingertips
                            color = (0, 255, 0)
                            radius = 4
                        elif i == 0:  # Wrist
                            color = (255, 0, 0)
                            radius = 6
                        else:
                            color = (255, 255, 0)
                            radius = 2
                            
                        cv2.circle(frame, (x, y), radius, color, -1)
                        
        except Exception as e:
            pass
            
    def _draw_help_panel(self, frame: np.ndarray):
        """Draw help panel with controls."""
        try:
            h, w = frame.shape[:2]
            
            # Help panel background
            panel_w, panel_h = 300, 200
            start_x = w - panel_w - 10
            start_y = h - panel_h - 10
            
            cv2.rectangle(frame, (start_x, start_y), 
                         (start_x + panel_w, start_y + panel_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (start_x, start_y), 
                         (start_x + panel_w, start_y + panel_h), (255, 255, 255), 2)
            
            # Help text
            help_lines = [
                "Controls:",
                "'s' - Static mode",
                "'d' - Dynamic mode",
                "'a' - Auto mode",
                "'l' - Toggle landmarks",
                "'c' - Toggle confidence",
                "'h' - Toggle help",
                "'q' - Quit"
            ]
            
            for i, line in enumerate(help_lines):
                y = start_y + 20 + i * 20
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                cv2.putText(frame, line, (start_x + 10, y), self.font, 0.5, color, 1)
                
        except Exception as e:
            pass


def main():
    """Main function to run enhanced LESSA demo."""
    demo = EnhancedLESSADemo()
    
    if demo.initialize_systems():
        demo.run_demo()
    else:
        print("❌ Failed to initialize recognition systems")
        print("Make sure you have:")
        print("  • Static training data: lessa_alphabet_data.json")
        print("  • Dynamic training data: lessa_dynamic_data.json (optional)")
        print("  • TensorFlow installed for dynamic recognition")


if __name__ == "__main__":
    main()