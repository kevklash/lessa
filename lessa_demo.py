"""
Enhanced demo application using holistic detection for LESSA.
Demonstrates full body pose, hand, and face detection for comprehensive sign language analysis.
"""

import cv2
import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.camera import Camera
from src.detection.holistic_detector import HolisticDetector
from src.detection.holistic_feature_extractor import HolisticFeatureExtractor

def main():
    """Main demo function for holistic LESSA detection."""
    print("🇸🇻 Starting LESSA (El Salvador Sign Language System) Demo...")
    print("🎯 Full body detection: Hands + Pose + Face")
    print("Press 'q' to quit, 'f' to show features, 'h' to toggle help, 's' to save sample")
    
    # Initialize components
    camera = Camera()
    holistic_detector = HolisticDetector()
    feature_extractor = HolisticFeatureExtractor()
    
    # Demo state
    show_features = False
    show_help = True
    sample_count = 0
    
    try:
        # Start camera
        if not camera.start():
            print("❌ Failed to start camera. Please check your webcam connection.")
            return
        
        print("📹 Camera started successfully!")
        print("🔍 Initializing holistic detection (hands + pose + face)...")
        
        while True:
            # Read frame from camera
            ret, frame = camera.read_frame()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Perform holistic detection
            annotated_frame, detection_data = holistic_detector.detect_holistic(frame)
            
            # Add information overlay
            info_frame = holistic_detector.draw_info(annotated_frame, detection_data)
            
            # Show features if requested
            if show_features and holistic_detector.is_detection_complete(detection_data):
                features = feature_extractor.extract_features(detection_data)
                info_frame = draw_feature_overlay(info_frame, features, detection_data)
            
            # Show help text
            if show_help:
                info_frame = draw_help_overlay(info_frame)
            
            # Add LESSA branding
            info_frame = add_lessa_branding(info_frame)
            
            # Display the frame
            cv2.imshow('LESSA - El Salvador Sign Language System', info_frame)
            
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
            elif key == ord('s'):
                # Save sample for testing
                if holistic_detector.is_detection_complete(detection_data):
                    save_detection_sample(detection_data, feature_extractor, sample_count)
                    sample_count += 1
                else:
                    print("⚠️  Cannot save sample - insufficient detection data")
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        camera.stop()
        holistic_detector.close()
        cv2.destroyAllWindows()
        print("🎯 LESSA demo ended. Resources cleaned up.")

def draw_feature_overlay(frame: np.ndarray, features: dict, detection_data: dict) -> np.ndarray:
    """Draw feature information overlay."""
    overlay_frame = frame.copy()
    
    # Create semi-transparent overlay - adjusted size and position
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
    
    # Hand features
    hand_features = features['hand_features']
    
    if hand_features['left_hand']:
        cv2.putText(overlay_frame, "Left Hand:", 
                   (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20
        
        # Hand shape info
        shape_data = hand_features['left_hand']['hand_shape']
        if shape_data:
            span_text = f"  Span: {shape_data.get('hand_span', 0):.3f}"
            curl_text = f"  Curl: {shape_data.get('average_finger_curl', 0):.3f}"
            cv2.putText(overlay_frame, span_text, 
                       (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
            cv2.putText(overlay_frame, curl_text, 
                       (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
    
    if hand_features['right_hand']:
        cv2.putText(overlay_frame, "Right Hand:", 
                   (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20
        
        # Hand shape info
        shape_data = hand_features['right_hand']['hand_shape']
        if shape_data:
            span_text = f"  Span: {shape_data.get('hand_span', 0):.3f}"
            curl_text = f"  Curl: {shape_data.get('average_finger_curl', 0):.3f}"
            cv2.putText(overlay_frame, span_text, 
                       (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
            cv2.putText(overlay_frame, curl_text, 
                       (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
    
    # Spatial relationships
    spatial_data = features['spatial_relationships']
    if 'hands_to_body' in spatial_data:
        cv2.putText(overlay_frame, "Hand Position:", 
                   (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        
        for hand_side in ['left_hand_to_body', 'right_hand_to_body']:
            hand_data = spatial_data['hands_to_body'].get(hand_side)
            if hand_data:
                height_level = hand_data.get('height_level', 'unknown')
                distance = hand_data.get('distance_to_torso', 0)
                
                side_name = hand_side.split('_')[0].capitalize()
                pos_text = f"  {side_name}: {height_level}"
                dist_text = f"    Dist: {distance:.3f}"
                
                cv2.putText(overlay_frame, pos_text, 
                           (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
                cv2.putText(overlay_frame, dist_text, 
                           (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
    
    return overlay_frame

def draw_help_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw help information overlay."""
    help_frame = frame.copy()
    
    # Create semi-transparent background - better positioning
    overlay = help_frame.copy()
    help_width = 320
    help_height = 190
    start_x = 10
    start_y = frame.shape[0] - help_height - 10
    
    cv2.rectangle(overlay, (start_x, start_y), (start_x + help_width, start_y + help_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, help_frame, 0.25, 0, help_frame)
    
    # Help text
    help_texts = [
        "LESSA Controls:",
        "Q - Quit demo",
        "F - Toggle features",
        "H - Toggle help",
        "S - Save sample",
        "",
        "Detection Status:",
        "Green = Good detection",
        "Red = Poor detection"
    ]
    
    y_start = start_y + 25
    for i, text in enumerate(help_texts):
        color = (0, 255, 255) if i == 0 or text.startswith("Detection") else (255, 255, 255)
        font_size = 0.5 if i == 0 or text.startswith("Detection") else 0.4
        
        cv2.putText(help_frame, text, 
                   (start_x + 15, y_start + i * 18), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
    
    return help_frame

def add_lessa_branding(frame: np.ndarray) -> np.ndarray:
    """Add LESSA branding to the frame."""
    branded_frame = frame.copy()
    
    # Add El Salvador flag colors accent - better positioning
    flag_height = 6
    flag_width = 150
    start_x = frame.shape[1] - flag_width - 15
    start_y = frame.shape[0] - 35
    
    # Blue stripe
    cv2.rectangle(branded_frame, (start_x, start_y), (start_x + flag_width, start_y + flag_height), (255, 0, 0), -1)
    # White stripe  
    cv2.rectangle(branded_frame, (start_x, start_y + flag_height), (start_x + flag_width, start_y + 2*flag_height), (255, 255, 255), -1)
    # Blue stripe
    cv2.rectangle(branded_frame, (start_x, start_y + 2*flag_height), (start_x + flag_width, start_y + 3*flag_height), (255, 0, 0), -1)
    
    # Add LESSA text
    cv2.putText(branded_frame, "El Salvador Sign Language", 
               (start_x, start_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return branded_frame

def save_detection_sample(detection_data: dict, feature_extractor: HolisticFeatureExtractor, sample_id: int):
    """Save a detection sample for analysis."""
    try:
        import json
        from datetime import datetime
        
        # Extract features
        features = feature_extractor.extract_features(detection_data)
        
        # Create sample data
        sample_data = {
            'sample_id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'detection_data': {
                # Serialize numpy arrays to lists for JSON compatibility
                'hands': {
                    'left_hand': _serialize_hand_data(detection_data['hands']['left_hand']),
                    'right_hand': _serialize_hand_data(detection_data['hands']['right_hand'])
                },
                'pose': _serialize_landmark_data(detection_data['pose']),
                'face': _serialize_landmark_data(detection_data['face']),
                'detection_confidence': detection_data['detection_confidence']
            },
            'features_summary': {
                'hands_detected': len([h for h in [detection_data['hands']['left_hand'], 
                                                  detection_data['hands']['right_hand']] if h]),
                'pose_detected': detection_data['pose'] is not None,
                'face_detected': detection_data['face'] is not None
            }
        }
        
        # Save to file
        os.makedirs('lessa_samples', exist_ok=True)
        filename = f'lessa_samples/sample_{sample_id:04d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"💾 Saved LESSA sample: {filename}")
        
    except Exception as e:
        print(f"❌ Error saving sample: {e}")

def _serialize_hand_data(hand_data):
    """Serialize hand data for JSON storage."""
    if not hand_data:
        return None
    
    return {
        'type': hand_data['type'],
        'landmarks': [[float(coord) for coord in landmark] for landmark in hand_data['landmarks']],
        'landmark_count': hand_data['landmark_count']
    }

def _serialize_landmark_data(landmark_data):
    """Serialize landmark data for JSON storage."""
    if not landmark_data:
        return None
    
    return {
        'type': landmark_data['type'],
        'landmarks': [[float(coord) for coord in landmark] for landmark in landmark_data['landmarks']],
        'landmark_count': landmark_data['landmark_count']
    }

if __name__ == "__main__":
    main()