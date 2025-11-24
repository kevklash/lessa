"""
Setup script for the Sign Language Translator project.
"""

from setuptools import setup, find_packages

setup(
    name="sign-language-translator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A real-time sign language recognition application using computer vision",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sign-language-translator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slt-demo=demo:main",
            "slt-collect=src.data.collector:main",
        ],
    },
)