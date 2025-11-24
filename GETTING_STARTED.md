# Getting Started with Sign Language Translator

This guide will help you set up and run your sign language translator project.

## Prerequisites

- Python 3.8 or higher
- Webcam (built-in or external)
- Windows, macOS, or Linux

## Installation

### Step 1: Clone or Navigate to Project Directory
Make sure you're in the project directory:
```bash
cd sign-language-translator
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

If you encounter any issues, try installing packages individually:
```bash
pip install opencv-python mediapipe numpy pandas tensorflow scikit-learn streamlit matplotlib seaborn pillow pyyaml tqdm
```

### Step 4: Verify Installation
Test your camera and hand detection:
```bash
python demo.py
```

This should open a window showing your webcam feed with hand detection. If you see your hands outlined with landmarks, everything is working correctly!

## Quick Start Guide

### 1. Test Hand Detection
Run the demo to verify everything works:
```bash
python demo.py
```

**Controls:**
- `Q` - Quit
- `F` - Toggle feature display
- `H` - Toggle help

### 2. Collect Training Data
Start collecting data for your first gesture (ASL alphabet):
```bash
python src/data/collector.py
```

**Controls:**
- `SPACE` - Start/Stop recording
- `N` - Next gesture
- `P` - Previous gesture
- `R` - Reset current gesture samples
- `Q` - Quit

**Data Collection Tips:**
- Position your hand clearly in view
- Use consistent lighting
- Vary hand positions slightly for better training data
- Collect at least 100 samples per gesture
- Keep background relatively clean

### 3. Training Your First Model
Once you have collected data for a few letters (A, B, C), you can train a basic model:

```bash
# This will be implemented in Phase 3
python src/models/static_model.py
```

## Project Structure Overview

```
sign-language-translator/
├── src/
│   ├── data/           # Data collection and preprocessing
│   ├── detection/      # Hand detection and feature extraction
│   ├── models/         # Machine learning models
│   ├── ui/            # User interfaces
│   └── utils/         # Utilities and configuration
├── data/
│   ├── raw/           # Raw collected data
│   ├── processed/     # Preprocessed data
│   └── models/        # Saved trained models
├── demo.py            # Quick demo application
├── config.yaml        # Configuration settings
└── roadmap.md         # Detailed development plan
```

## Configuration

Edit `config.yaml` to customize:
- Camera settings (resolution, FPS)
- MediaPipe parameters
- Data collection settings
- Gestures to recognize

## Troubleshooting

### Camera Issues
- **Camera not found**: Check if your webcam is connected and not being used by another application
- **Poor detection**: Ensure good lighting and clear background
- **Wrong camera**: Change `device_id` in `config.yaml` (try 0, 1, 2...)

### Installation Issues
- **OpenCV not working**: Try `pip install opencv-python-headless` instead
- **MediaPipe issues**: Ensure you have Python 3.8+ and try reinstalling: `pip uninstall mediapipe && pip install mediapipe`
- **TensorFlow warnings**: These are usually safe to ignore for CPU usage

### Performance Issues
- **Slow detection**: Lower camera resolution in `config.yaml`
- **High CPU usage**: Reduce `fps` setting in configuration
- **Memory issues**: Close other applications and restart

## Next Steps

1. **Collect Data**: Start with 3-5 ASL letters (A, B, C, D, E)
2. **Train Model**: Follow Phase 3 of the roadmap
3. **Test Recognition**: Evaluate your trained model
4. **Expand Vocabulary**: Add more gestures gradually
5. **Improve Accuracy**: Collect more diverse training data

## Getting Help

- Check the `roadmap.md` for detailed development phases
- Review configuration in `config.yaml`
- Look at example data in `data/raw/` after collection
- Check the console output for error messages

## Development Phases

This project follows a structured development approach:

1. **Phase 1**: Setup and Data Collection ← *You are here*
2. **Phase 2**: Feature Extraction and Preprocessing
3. **Phase 3**: Static Gesture Recognition (A-Z)
4. **Phase 4**: Dynamic Gesture Recognition
5. **Phase 5**: Real-time Application
6. **Phase 6**: Enhancement and Deployment

Follow the roadmap for detailed implementation guidance!

---

Happy coding! 🚀 You're building something amazing that can help bridge communication barriers.