# Sign Language Translator

A real-time sign language recognition application using computer vision and machine learning.

## Project Structure

```
sign-language-translator/
├── src/
│   ├── data/
│   │   ├── collector.py      # Data collection utilities
│   │   ├── preprocessor.py   # Data preprocessing pipeline
│   │   └── augmentation.py   # Data augmentation techniques
│   ├── models/
│   │   ├── static_model.py   # Static gesture recognition model
│   │   ├── dynamic_model.py  # Dynamic gesture recognition model
│   │   └── utils.py          # Model utilities
│   ├── detection/
│   │   ├── hand_detector.py  # MediaPipe hand detection
│   │   └── feature_extractor.py  # Feature extraction from landmarks
│   ├── ui/
│   │   ├── streamlit_app.py  # Streamlit web interface
│   │   └── tkinter_app.py    # Desktop application
│   └── utils/
│       ├── camera.py         # Camera utilities
│       ├── config.py         # Configuration settings
│       └── helpers.py        # General helper functions
├── data/
│   ├── raw/                  # Raw collected data
│   ├── processed/            # Preprocessed data
│   └── models/               # Saved trained models
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── tests/
├── requirements.txt
├── setup.py
├── config.yaml
└── README.md
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the data collection tool:
   ```bash
   python src/data/collector.py
   ```

3. Train a basic model:
   ```bash
   python src/models/static_model.py
   ```

4. Start the real-time application:
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

## Features

- [x] Real-time hand detection using MediaPipe
- [x] Data collection interface
- [ ] Static gesture recognition (A-Z)
- [ ] Dynamic gesture recognition
- [ ] Web-based UI
- [ ] Desktop application

## Requirements

- Python 3.8+
- Webcam
- GPU recommended for training (optional)

## Documentation

See [roadmap.md](roadmap.md) for detailed development plan and architecture.

## License

MIT License