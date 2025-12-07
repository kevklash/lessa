"""
Dynamic gesture training pipeline for LESSA.
Helps users collect dynamic gesture data and train the LSTM model.
"""

import sys
from pathlib import Path
from typing import Optional

from src.data.dynamic_collector import DynamicGestureCollector
from dynamic_gesture_recognizer import DynamicGestureRecognizer


def main():
    """Main training pipeline."""
    print("🎯 LESSA Dynamic Gesture Training Pipeline")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Collect dynamic gesture samples")
        print("2. Train dynamic recognition model")  
        print("3. Test existing model")
        print("4. View training data statistics")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            collect_samples()
        elif choice == "2":
            train_model()
        elif choice == "3":
            test_model()
        elif choice == "4":
            show_statistics()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")


def collect_samples():
    """Collect dynamic gesture samples."""
    print("\n📹 Dynamic Gesture Sample Collection")
    print("-" * 40)
    
    collector = DynamicGestureCollector()
    
    # Show current data status
    print("Current sample counts:")
    for letter in collector.dynamic_letters:
        count = len(collector.data.get(letter, []))
        print(f"  • {letter}: {count} samples")
        
    print("\nRecommended: 10-20 samples per letter for good training")
    
    proceed = input("\nProceed with collection? (y/n): ").lower().strip()
    if proceed == 'y':
        collector.collect_samples_interactive()
    else:
        print("📋 Collection cancelled")


def train_model():
    """Train the dynamic recognition model."""
    print("\n🧠 Dynamic Model Training")
    print("-" * 30)
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} detected")
    except ImportError:
        print("❌ TensorFlow not found!")
        print("Install with: pip install tensorflow")
        return
        
    # Check training data
    data_file = Path("lessa_dynamic_data.json")
    if not data_file.exists():
        print("❌ No training data found!")
        print("Collect samples first using option 1.")
        return
        
    recognizer = DynamicGestureRecognizer()
    
    print("🔄 Starting model training...")
    
    if recognizer.train_model():
        print("\n✅ Training completed successfully!")
        
        # Show evaluation results
        evaluation = recognizer.evaluate_model()
        if evaluation:
            print(f"\n📊 Training Results:")
            print(f"   • Overall accuracy: {evaluation['overall_accuracy']:.1%}")
            print(f"   • Total samples: {evaluation['total_samples']}")
            
            print("\n📈 Per-letter accuracy:")
            for letter, acc in evaluation['class_accuracies'].items():
                print(f"   • {letter}: {acc:.1%}")
                
        print(f"\n💾 Model saved for use in recognition demo")
        
    else:
        print("❌ Training failed!")
        print("Check that you have sufficient samples for each letter")


def test_model():
    """Test existing trained model."""
    print("\n🧪 Model Testing")
    print("-" * 20)
    
    recognizer = DynamicGestureRecognizer()
    
    # Check if model exists
    info = recognizer.get_model_info()
    
    if not info['model_available']:
        print("❌ No trained model found!")
        print("Train a model first using option 2.")
        return
        
    if not info['tensorflow_available']:
        print("❌ TensorFlow not available for testing!")
        return
        
    print("✅ Model loaded successfully")
    print(f"\n🔧 Model Information:")
    print(f"   • Dynamic letters: {', '.join(info['dynamic_letters'])}")
    print(f"   • Confidence threshold: {info['confidence_threshold']}")
    print(f"   • Sequence length: {info['sequence_length']} frames")
    
    if 'total_parameters' in info:
        print(f"   • Model parameters: {info['total_parameters']:,}")
        print(f"   • Input shape: {info['input_shape']}")
        print(f"   • Output shape: {info['output_shape']}")
        
    # Run evaluation if training data exists
    evaluation = recognizer.evaluate_model()
    if evaluation:
        print(f"\n📊 Model Performance:")
        print(f"   • Overall accuracy: {evaluation['overall_accuracy']:.1%}")
        print(f"   • Evaluated on: {evaluation['total_samples']} samples")
        
        print("\n📈 Per-letter performance:")
        for letter, acc in evaluation['class_accuracies'].items():
            print(f"   • {letter}: {acc:.1%}")
    else:
        print("\n⚠️  Could not evaluate model (no training data)")
        
    print(f"\n✅ Model is ready for use in enhanced demo!")


def show_statistics():
    """Show training data statistics."""
    print("\n📊 Training Data Statistics")
    print("-" * 30)
    
    data_file = Path("lessa_dynamic_data.json")
    
    if not data_file.exists():
        print("❌ No training data found!")
        print("Collect samples first using option 1.")
        return
        
    import json
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        print("📈 Sample counts by letter:")
        total_samples = 0
        
        for letter in ['J', 'Z']:
            count = len(data.get(letter, []))
            total_samples += count
            
            status = "✅" if count >= 10 else "⚠️ " if count >= 5 else "❌"
            print(f"   {status} {letter}: {count} samples")
            
        print(f"\n📋 Total samples: {total_samples}")
        
        if total_samples >= 20:
            print("✅ Sufficient data for training!")
        elif total_samples >= 10:
            print("⚠️  Minimal data - consider collecting more samples")
        else:
            print("❌ Insufficient data for training")
            print("   Recommend: 10+ samples per letter")
            
        # Show sequence length statistics
        if total_samples > 0:
            sequence_lengths = []
            motion_profiles = []
            
            for letter_data in data.values():
                for sample in letter_data:
                    if 'sequence_length' in sample:
                        sequence_lengths.append(sample['sequence_length'])
                    if 'motion_profile' in sample:
                        motion_profiles.append(sample['motion_profile'])
                        
            if sequence_lengths:
                import numpy as np
                print(f"\n📏 Sequence length statistics:")
                print(f"   • Average: {np.mean(sequence_lengths):.1f} frames")
                print(f"   • Range: {min(sequence_lengths)}-{max(sequence_lengths)} frames")
                
    except Exception as e:
        print(f"❌ Error reading data: {e}")


if __name__ == "__main__":
    main()