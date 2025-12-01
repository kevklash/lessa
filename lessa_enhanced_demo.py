"""
Enhanced LESSA Demo with Advanced Camera Management
Includes camera detection, quality assessment, and optimization for optimal sign language recognition.
"""

import cv2
import sys
import os
import numpy as np
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.enhanced_camera import EnhancedCamera, CameraManager
from src.detection.holistic_detector import HolisticDetector
from src.detection.holistic_feature_extractor import HolisticFeatureExtractor

def main():
    """Main demo function with enhanced camera management."""
    print("🇸🇻 LESSA Enhanced Demo - Camera Management & Quality Assessment")
    print("=" * 60)
    
    # Detect available cameras
    available_cameras = CameraManager.detect_cameras()
    
    if not available_cameras:
        print("❌ No cameras detected. Please connect a camera and try again.")
        return
    
    # Display available cameras
    print(CameraManager.format_camera_list(available_cameras))
    
    # Camera selection
    selected_camera = select_camera(available_cameras)
    if selected_camera is None:
        return
    
    print(f"📹 Using Camera {selected_camera.device_id}")
    
    # Initialize components
    camera = EnhancedCamera(selected_camera.device_id)
    holistic_detector = HolisticDetector()
    feature_extractor = HolisticFeatureExtractor()
    
    # Demo state
    show_features = False
    show_help = True
    show_quality = False
    sample_count = 0
    performance_monitor = PerformanceMonitor()
    
    print("\n🎯 Controls:")
    print("Q - Quit | F - Features | H - Help | S - Save | C - Camera Quality | P - Performance")
    
    try:
        # Start camera with quality assessment
        print("\n🚀 Starting camera with quality optimization...")
        if not camera.start(optimize_quality=True):
            print("❌ Failed to start camera")
            return
        
        # Display quality report
        print(camera.get_quality_report())
        
        print("\n▶️  Demo started! Use controls to interact.")
        
        while True:
            frame_start_time = time.time()
            
            # Read frame
            ret, frame = camera.read_frame()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Perform holistic detection
            detection_start = time.time()
            annotated_frame, detection_data = holistic_detector.detect_holistic(frame)
            detection_time = (time.time() - detection_start) * 1000
            
            # Add information overlay
            info_frame = holistic_detector.draw_info(annotated_frame, detection_data)
            
            # Show features if requested
            if show_features and holistic_detector.is_detection_complete(detection_data):
                features = feature_extractor.extract_features(detection_data)
                info_frame = draw_feature_overlay(info_frame, features, detection_data)
            
            # Show quality info if requested
            if show_quality and camera.current_quality:
                info_frame = draw_quality_overlay(info_frame, camera.current_quality)
            
            # Show help text
            if show_help:
                info_frame = draw_enhanced_help_overlay(info_frame)
            
            # Add performance monitor
            frame_time = (time.time() - frame_start_time) * 1000
            performance_monitor.update(detection_time, frame_time)
            info_frame = draw_performance_overlay(info_frame, performance_monitor)
            
            # Add LESSA branding
            info_frame = add_lessa_branding(info_frame)
            
            # Display the frame
            cv2.imshow('LESSA Enhanced - Camera Management Demo', info_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                show_features = not show_features
                print(f"🔬 Feature display: {'ON' if show_features else 'OFF'}")
            elif key == ord('h'):
                show_help = not show_help
                print(f"💡 Help display: {'ON' if show_help else 'OFF'}")
            elif key == ord('c'):
                show_quality = not show_quality
                print(f"📊 Quality display: {'ON' if show_quality else 'OFF'}")
            elif key == ord('p'):
                print(performance_monitor.get_report())
            elif key == ord('s'):
                if holistic_detector.is_detection_complete(detection_data):
                    save_detection_sample(detection_data, feature_extractor, sample_count, camera.current_quality)
                    sample_count += 1
                else:
                    print("⚠️  Cannot save sample - insufficient detection data")
            elif key == ord('r'):
                # Reassess camera quality
                print("🔄 Reassessing camera quality...")
                camera.current_quality = camera.assess_quality()
                print(camera.get_quality_report())
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        camera.stop()
        holistic_detector.close()
        cv2.destroyAllWindows()
        print("🎯 LESSA Enhanced Demo ended. Resources cleaned up.")

def select_camera(cameras):
    """Interactive camera selection."""
    if len(cameras) == 1:
        print(f"📹 Auto-selecting best camera: Camera {cameras[0].device_id}")
        return cameras[0]
    
    print("\n📹 Multiple cameras detected. Please select:")
    print("=" * 40)
    
    for i, cam in enumerate(cameras):
        quality_emoji = "🟢" if cam.quality_score >= 80 else "🟡" if cam.quality_score >= 60 else "🔴"
        print(f"{i + 1}. {quality_emoji} Camera {cam.device_id}: {cam.max_width}x{cam.max_height} @ {cam.max_fps:.1f}fps")
        print(f"    Quality: {cam.quality_score:.1f}/100, Latency: {cam.latency_ms:.1f}ms")
    
    print("0. Auto-select best camera")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-{}): ".format(len(cameras)))
            choice = int(choice)
            
            if choice == 0:
                best_camera = max(cameras, key=lambda x: x.quality_score)
                print(f"📹 Auto-selected: Camera {best_camera.device_id} (Quality: {best_camera.quality_score:.1f})")
                return best_camera
            elif 1 <= choice <= len(cameras):
                selected = cameras[choice - 1]
                print(f"📹 Selected: Camera {selected.device_id}")
                return selected
            else:
                print("❌ Invalid selection. Please try again.")
        
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input or cancelled.")
            return None

class PerformanceMonitor:
    """Monitor real-time performance metrics."""
    
    def __init__(self):
        self.detection_times = []
        self.frame_times = []
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def update(self, detection_time: float, frame_time: float):
        """Update performance metrics."""
        self.detection_times.append(detection_time)
        self.frame_times.append(frame_time)
        
        # Keep only last 30 measurements
        if len(self.detection_times) > 30:
            self.detection_times.pop(0)
            self.frame_times.pop(0)
        
        # Calculate FPS
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def get_avg_detection_time(self) -> float:
        """Get average detection time."""
        return np.mean(self.detection_times) if self.detection_times else 0
    
    def get_avg_frame_time(self) -> float:
        """Get average frame time."""
        return np.mean(self.frame_times) if self.frame_times else 0
    
    def get_report(self) -> str:
        """Get performance report."""
        return f"""
🚀 LESSA Performance Report:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Current FPS:       {self.current_fps}
⚡ Avg Detection:     {self.get_avg_detection_time():.1f}ms
🎯 Avg Frame Time:    {self.get_avg_frame_time():.1f}ms
📈 Performance:       {'🟢 Excellent' if self.current_fps >= 25 else '🟡 Good' if self.current_fps >= 15 else '🔴 Poor'}
"""

def draw_quality_overlay(frame: np.ndarray, quality) -> np.ndarray:
    """Draw camera quality overlay."""
    overlay_frame = frame.copy()
    
    # Semi-transparent background
    overlay = overlay_frame.copy()
    width, height = 300, 200
    start_x = 10
    start_y = 100
    
    cv2.rectangle(overlay, (start_x, start_y), (start_x + width, start_y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, overlay_frame, 0.25, 0, overlay_frame)
    
    # Quality metrics
    y_offset = start_y + 25
    cv2.putText(overlay_frame, "Camera Quality", 
               (start_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_offset += 25
    metrics = [
        f"Overall: {quality.overall_score:.0f}/100",
        f"Resolution: {quality.resolution_score:.0f}/100",
        f"Latency: {quality.latency_score:.0f}/100",
        f"Brightness: {quality.brightness_score:.0f}/100",
        f"Sharpness: {quality.sharpness_score:.0f}/100"
    ]
    
    for metric in metrics:
        cv2.putText(overlay_frame, metric, 
                   (start_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
    
    return overlay_frame

def draw_performance_overlay(frame: np.ndarray, monitor: PerformanceMonitor) -> np.ndarray:
    """Draw performance overlay."""
    overlay_frame = frame.copy()
    
    # Performance indicators in top-left corner
    fps_color = (0, 255, 0) if monitor.current_fps >= 25 else (0, 255, 255) if monitor.current_fps >= 15 else (0, 0, 255)
    
    cv2.putText(overlay_frame, f"FPS: {monitor.current_fps}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
    
    detection_time = monitor.get_avg_detection_time()
    detection_color = (0, 255, 0) if detection_time <= 30 else (0, 255, 255) if detection_time <= 50 else (0, 0, 255)
    
    cv2.putText(overlay_frame, f"Detection: {detection_time:.1f}ms", 
               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color, 1)
    
    return overlay_frame

def draw_enhanced_help_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw enhanced help overlay with camera controls."""
    help_frame = frame.copy()
    
    # Background
    overlay = help_frame.copy()
    help_width = 350
    help_height = 220
    start_x = 10
    start_y = frame.shape[0] - help_height - 10
    
    cv2.rectangle(overlay, (start_x, start_y), (start_x + help_width, start_y + help_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, help_frame, 0.25, 0, help_frame)
    
    # Help text
    help_texts = [
        "LESSA Enhanced Controls:",
        "Q - Quit demo",
        "F - Toggle features",
        "H - Toggle help",
        "S - Save sample",
        "C - Toggle quality info",
        "P - Show performance",
        "R - Reassess quality",
        "",
        "Quality Indicators:",
        "🟢 Excellent  🟡 Good  🔴 Poor"
    ]
    
    y_start = start_y + 25
    for i, text in enumerate(help_texts):
        if i == 0 or text.startswith("Quality"):
            color = (0, 255, 255)
            font_size = 0.5
        else:
            color = (255, 255, 255)
            font_size = 0.4
        
        cv2.putText(help_frame, text, 
                   (start_x + 15, y_start + i * 18), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
    
    return help_frame

def draw_feature_overlay(frame: np.ndarray, features: dict, detection_data: dict) -> np.ndarray:
    """Draw feature overlay (reuse from previous version)."""
    # Same implementation as before
    overlay_frame = frame.copy()
    
    # Create semi-transparent overlay
    overlay = overlay_frame.copy()
    overlay_width = 380
    overlay_height = min(450, frame.shape[0] - 20)
    start_x = frame.shape[1] - overlay_width - 10
    start_y = 10
    
    cv2.rectangle(overlay, (start_x, start_y), (start_x + overlay_width, start_y + overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, overlay_frame, 0.25, 0, overlay_frame)
    
    y_offset = start_y + 25
    x_start = start_x + 15
    
    # Title
    cv2.putText(overlay_frame, "LESSA Features", 
               (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_offset += 35
    
    # Hand features summary
    hand_features = features['hand_features']
    
    if hand_features['left_hand'] or hand_features['right_hand']:
        hands_detected = []
        if hand_features['left_hand']:
            hands_detected.append("Left")
        if hand_features['right_hand']:
            hands_detected.append("Right")
        
        cv2.putText(overlay_frame, f"Hands: {', '.join(hands_detected)}", 
                   (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
    
    # Detection summary
    summary = {
        'Pose': features['pose_features'] is not None,
        'Face': features['face_features'] is not None
    }
    
    for component, detected in summary.items():
        color = (0, 255, 0) if detected else (128, 128, 128)
        status = "✓" if detected else "✗"
        cv2.putText(overlay_frame, f"{component}: {status}", 
                   (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
    
    return overlay_frame

def add_lessa_branding(frame: np.ndarray) -> np.ndarray:
    """Add LESSA branding (reuse from previous version)."""
    branded_frame = frame.copy()
    
    flag_height = 6
    flag_width = 180
    start_x = frame.shape[1] - flag_width - 15
    start_y = frame.shape[0] - 35
    
    # El Salvador flag
    cv2.rectangle(branded_frame, (start_x, start_y), (start_x + flag_width, start_y + flag_height), (255, 0, 0), -1)
    cv2.rectangle(branded_frame, (start_x, start_y + flag_height), (start_x + flag_width, start_y + 2*flag_height), (255, 255, 255), -1)
    cv2.rectangle(branded_frame, (start_x, start_y + 2*flag_height), (start_x + flag_width, start_y + 3*flag_height), (255, 0, 0), -1)
    
    cv2.putText(branded_frame, "LESSA - Enhanced Camera Demo", 
               (start_x, start_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return branded_frame

def save_detection_sample(detection_data: dict, feature_extractor, sample_id: int, quality_info):
    """Save detection sample with camera quality info."""
    try:
        import json
        from datetime import datetime
        
        features = feature_extractor.extract_features(detection_data)
        
        sample_data = {
            'sample_id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'camera_quality': {
                'overall_score': quality_info.overall_score if quality_info else 0,
                'resolution_score': quality_info.resolution_score if quality_info else 0,
                'sharpness_score': quality_info.sharpness_score if quality_info else 0
            },
            'detection_summary': {
                'hands_detected': len([h for h in [detection_data['hands']['left_hand'], 
                                                  detection_data['hands']['right_hand']] if h]),
                'pose_detected': detection_data['pose'] is not None,
                'face_detected': detection_data['face'] is not None
            }
        }
        
        os.makedirs('lessa_samples', exist_ok=True)
        filename = f'lessa_samples/enhanced_sample_{sample_id:04d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"💾 Saved enhanced LESSA sample: {filename}")
        
    except Exception as e:
        print(f"❌ Error saving sample: {e}")

if __name__ == "__main__":
    main()