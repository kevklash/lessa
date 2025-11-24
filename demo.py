"""
Simple demo application to test hand detection and feature extraction.
Run this to verify that your setup is working correctly.
"""

import cv2
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.camera import Camera
from src.detection.hand_detector import HandDetector
from src.detection.feature_extractor import FeatureExtractor

def main():
    """Main demo function."""
    print("Starting Sign Language Translator Demo...")
    print("Press 'q' to quit, 'f' to show features, 'h' to toggle help")
    
    # Initialize components
    camera = Camera()
    hand_detector = HandDetector()
    feature_extractor = FeatureExtractor()
    
    # Demo state
    show_features = False
    show_help = True
    
    try:
        # Start camera
        if not camera.start():
            print("Failed to start camera. Please check your webcam connection.")
            return
        
        print("Camera started successfully!")
        print("Initializing hand detection...")
        
        while True:
            # Read frame from camera
            ret, frame = camera.read_frame()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Detect hands
            annotated_frame, hands_data = hand_detector.detect_hands(frame)
            
            # Add information overlay
            info_frame = hand_detector.draw_info(annotated_frame, hands_data)
            
            # Show features if requested
            if show_features and hands_data:
                for i, hand_data in enumerate(hands_data):
                    features = feature_extractor.extract_features(hand_data)
                    
                    # Display some key features
                    y_start = 150 + i * 100
                    cv2.putText(info_frame, f"Hand {i+1} Features:", 
                               (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Show number of features
                    feature_count = len(features['raw_landmarks'])
                    cv2.putText(info_frame, f"Landmarks: {feature_count}", 
                               (10, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Show some distances
                    distances = features['distances'][:3]  # First 3 distances
                    distance_text = f"Distances: {[f'{d:.3f}' for d in distances]}"
                    cv2.putText(info_frame, distance_text, 
                               (10, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Show help text
            if show_help:
                help_texts = [
                    "Controls:",
                    "Q - Quit",
                    "F - Toggle Features", 
                    "H - Toggle Help"
                ]
                
                for i, text in enumerate(help_texts):
                    cv2.putText(info_frame, text, 
                               (info_frame.shape[1] - 200, 30 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display the frame
            cv2.imshow('Sign Language Translator Demo', info_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                show_features = not show_features
                print(f"Feature display: {'ON' if show_features else 'OFF'}")
            elif key == ord('h'):
                show_help = not show_help
                print(f"Help display: {'ON' if show_help else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        # Cleanup
        camera.stop()
        hand_detector.close()
        cv2.destroyAllWindows()
        print("Demo ended. Resources cleaned up.")

if __name__ == "__main__":
    main()