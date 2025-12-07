# Dynamic Gesture Recognition - Quick Start Guide

This guide will help you set up and use the dynamic gesture recognition system for LESSA letters J and Z.

## Prerequisites

1. **Basic LESSA setup** - Make sure static recognition is working
2. **TensorFlow installation** - Required for LSTM training and recognition

```bash
pip install tensorflow>=2.12.0
```

## Step 1: Collect Training Data

### Launch Collection Interface
```bash
python train_dynamic_gestures.py
```
Choose **option 1**: "Collect dynamic gesture samples"

### Collection Tips for Best Results

#### Letter J (Hook Motion)
- **Position**: Start with index finger pointing up
- **Movement**: Draw downward stroke, then curve left at bottom
- **Duration**: Complete motion in 0.5-1 second
- **Consistency**: Keep the hook curve smooth and consistent

#### Letter Z (Zigzag Motion)
- **Position**: Start with index finger at top-left
- **Movement**: Horizontal right → diagonal down-left → horizontal right
- **Duration**: Complete motion in 0.8-1.2 seconds
- **Corners**: Make distinct direction changes at each corner

### Recommended Sample Count
- **Minimum**: 10 samples per letter
- **Optimal**: 15-20 samples per letter
- **Quality over quantity**: Better to have fewer good samples than many inconsistent ones

## Step 2: Train Recognition Model

### Launch Training
```bash
python train_dynamic_gestures.py
```
Choose **option 2**: "Train dynamic recognition model"

### Training Process
1. **Feature Extraction** - Converts motion sequences to numerical features
2. **LSTM Training** - Deep learning model learns gesture patterns  
3. **Validation** - Model tested on held-out data
4. **Model Saving** - Trained model saved for recognition use

### Expected Training Time
- **CPU**: 2-5 minutes for small datasets
- **GPU**: 30 seconds - 2 minutes (if available)

### Success Indicators
- **Accuracy**: >85% overall accuracy is good, >90% is excellent
- **Per-letter accuracy**: Both J and Z should have >80% accuracy
- **Loss convergence**: Training loss should decrease steadily

## Step 3: Test Recognition

### Quick Model Test
```bash
python train_dynamic_gestures.py
```
Choose **option 3**: "Test existing model"

### Full Enhanced Demo
```bash
python lessa_enhanced_dynamic_demo.py
```

## Using the Enhanced Demo

### Controls
- **'a'** - Auto mode (recommended): Automatically switches between static/dynamic
- **'s'** - Manual static mode: Only static alphabet recognition
- **'d'** - Manual dynamic mode: Only dynamic gesture recognition
- **'l'** - Toggle landmark display
- **'c'** - Toggle confidence scores
- **'h'** - Toggle help panel

### Recognition Workflow

#### Auto Mode (Recommended)
1. **Static Recognition**: Hold hand still in letter shape
2. **Dynamic Detection**: System detects when you start moving
3. **Motion Recording**: Captures gesture sequence automatically
4. **Recognition**: Displays result when motion stops

#### Manual Mode
- **Static**: Traditional alphabet recognition for all letters except J/Z
- **Dynamic**: Perform J or Z gestures, system will attempt recognition

### Performance Tips

#### For Best Recognition Results
1. **Good Lighting**: Ensure hands are well-lit and visible
2. **Clear Background**: Minimal visual clutter behind hands  
3. **Consistent Speed**: Perform gestures at moderate, consistent speed
4. **Complete Motions**: Finish the complete letter motion before stopping
5. **Practice**: Try the same gesture multiple times to see consistency

#### Troubleshooting Common Issues

**"No gesture detected"**
- Motion may be too small or slow
- Ensure hand is visible throughout motion
- Try more pronounced movements

**"Low confidence recognition"**
- Motion may be inconsistent with training data
- Try matching the speed/style used during training
- Check if gesture is complete

**"Getting stuck in dynamic mode"**
- Hold hand completely still for 2-3 seconds
- System will return to static mode automatically

## Advanced Usage

### Training Data Management

#### View Data Statistics
```bash
python train_dynamic_gestures.py
```
Choose **option 4**: "View training data statistics"

#### Manual Data File Location
- **Training data**: `lessa_dynamic_data.json`
- **Trained model**: `models/dynamic/dynamic_gesture_model.h5`
- **Preprocessing**: `models/dynamic/dynamic_scaler.pkl` and `dynamic_encoder.pkl`

### Custom Training Parameters

Edit `dynamic_gesture_recognizer.py` to adjust:
- **Confidence threshold**: Lower = more sensitive, higher = more strict
- **Sequence length**: Longer = more motion data, shorter = faster response
- **Model architecture**: LSTM layers, neurons, dropout rates

### Integration with Other Systems

The dynamic recognition system can be imported and used in custom applications:

```python
from dynamic_gesture_recognizer import DynamicGestureRecognizer

# Initialize recognizer
recognizer = DynamicGestureRecognizer()

# Recognition on gesture sequence
prediction, confidence = recognizer.recognize_gesture(gesture_sequence)
```

## Next Steps

1. **Collect Quality Data** - Focus on consistent, clear gestures
2. **Train Initial Model** - Start with minimum viable dataset
3. **Test and Iterate** - Use enhanced demo to test real-world performance
4. **Refine Training Data** - Add more samples for letters with low accuracy
5. **Explore Advanced Features** - Experiment with different training parameters

## Support

For issues or questions:
1. Check training data quality and quantity first
2. Ensure TensorFlow is properly installed
3. Verify webcam and lighting conditions
4. Review error messages in terminal output

The system is designed to be robust and user-friendly, but like all machine learning systems, quality training data is essential for good performance!