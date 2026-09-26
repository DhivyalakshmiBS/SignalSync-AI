# ⚡ SignalSense AI – Power Quality Disturbance Detection

## 📌 Project Overview

SignalSense AI is a Machine Learning-based electrical waveform analysis system that detects and classifies different power quality disturbances from uploaded CSV waveform data.

The system extracts important characteristics from the electrical signal and uses a trained Machine Learning model to identify the corresponding disturbance type and severity level.

The project demonstrates the application of Machine Learning and signal analysis in electrical power quality monitoring.

---

## ✨ Features

- 📂 Upload electrical waveform in CSV format
- 📈 Electrical waveform visualization
- 🔍 Automatic waveform analysis
- ⚡ Power quality disturbance detection
- 🤖 Machine Learning-based classification
- 📊 Time-domain and frequency-domain feature extraction
- 🚦 Severity indication for detected disturbances
- 🖥️ Interactive web-based interface

---

## 🔧 Disturbance Types

- Flicker
- Flicker with Sag
- Flicker with Swell
- Harmonics
- Harmonics with Sag
- Harmonics with Swell
- Interruption
- Notch
- Oscillatory Transient
- Pure Sinusoidal
- Sag
- Sag with Harmonics
- Sag with Oscillatory Transient
- Swell
- Swell with Harmonics
- Swell with Oscillatory Transient
- Transient

---

## ⚙️ Working Process

1. Upload an electrical waveform CSV file.
2. Read and process the waveform samples.
3. Extract time-domain and frequency-domain features.
4. Pass the extracted features to the trained Machine Learning model.
5. Classify the power quality disturbance.
6. Determine the corresponding severity level.
7. Display the waveform and analysis result.

---

## 🧠 Concepts Used

- Machine Learning
- Power Quality Analysis
- Electrical Signal Processing
- Time-Domain Analysis
- Frequency-Domain Analysis
- Fast Fourier Transform (FFT)
- Feature Extraction
- Classification
- Data Processing
- Waveform Visualization

---

## 📊 Dataset

The project uses the **SEED Power Quality Disturbance Dataset** for training the Machine Learning model.

The dataset contains electrical waveform samples representing different power quality disturbance conditions.

---

## 💻 Technologies Used

- Programming Language: Python
- Machine Learning: Scikit-learn
- Web Framework: Streamlit
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib
- Model Handling: Joblib

---
