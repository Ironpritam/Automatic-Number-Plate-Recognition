# Automatic Number Plate Recognition (ANPR) System for Indian Vehicles

> A production-ready Computer Vision & Multi-Lingual Optical Character Recognition (OCR) pipeline leveraging fine-tuned **YOLOv5**, morphological image deskewing/enhancement, offline Devanagari transliteration, and a Flask RESTful API.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-00FFFF.svg)](https://github.com/ultralytics/yolov5)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](CODE/LICENSE)

---

## 📌 Executive Overview

In India's diverse traffic ecosystem, vehicle license plates vary significantly in font styles, aspect ratios, background colors, and regional language scripts (such as Devanagari/Marathi). Standard commercial OCR solutions often fail when deployed on non-standard, skewed, or dirty plates.

Developed as a **B.Tech Capstone Project** at **Ramrao Adik Institute of Technology (D.Y. Patil Deemed to be University)** , this project delivers an end-to-end Automatic Number Plate Recognition system specifically engineered for Indian traffic conditions.

The research behind this system has been peer-reviewed and published in the **International Conference on Advances in Computing and Communications (ICACC)** under the title:  
`"Automatic Number Plate Recognition System for Indian Number Plates using Machine Learning Techniques"`.

---

## 🚀 Key Features & Capabilities

### 🎯 1. High-Precision Bounding Box Detection (YOLOv5)
- **Custom Trained Detector**: Fine-tuned YOLOv5 model trained across 100+ epochs on custom-annotated Indian vehicle datasets (cars, motorcycles, auto-rickshaws, commercial trucks).
- **Small Object Localization**: Accurately localizes small, distant, or multi-angle license plate region of interest (ROI) bounding boxes in high-resolution video frames.

### 🔬 2. Robust Image Preprocessing & Deskewing
- **Projection Profile Deskewing**: Automatically estimates plate rotation angle using vertical projection profile variance analysis (`scipy.ndimage`) and rotates skewed plates back to horizontal alignment.
- **Noise Elimination & Contrast Enhancement**: Uses Gaussian Blurring, Bilateral Filtering (edges preserved), Otsu’s Adaptive Binary Binarization, and Morphological Dilation/Erosion to clean noisy plate crops.

### 🌐 3. Multi-Lingual OCR Engine (English + Devanagari Script)
- **Dual Script Recognition**: Supports standard English (Latin alphanumeric) and regional Marathi (Devanagari) plates.
- **Offline Devanagari Parser**: Features a pure Python transliteration mapping engine (`०-९` $\rightarrow$ `0-9` and state codes like `महाराष्ट्र` $\rightarrow$ `MH`) ensuring 100% offline accuracy without network dependencies.
- **Online Fallback**: Integrated fallback handling via `googletrans` API.

### ⚡ 4. Enterprise Architecture & REST API
- **Modular Package Structure (`src/`)**: Decouples image preprocessing (`preprocessing.py`), OCR engine (`ocr_engine.py`), YOLOv5 detector (`detector.py`), and logger (`logger.py`).
- **RESTful API Endpoint (`/api/v1/predict`)**: Accepts Base64 encoded images and returns structured JSON predictions with timestamps, confidence scores, and language metadata.
- **Audit Logging**: Maintains real-time detection logs in `data.csv` and JSON audit files.

---

## 🔄 End-to-End System Pipeline

```text
                  Input Vehicle Image / Stream Feed
                                 │
                                 ▼
              YOLOv5 License Plate Bounding Box Detection
                                 │
                                 ▼
                     Crop Region of Interest (ROI)
                                 │
                                 ▼
         Image Preprocessing & Projection Profile Deskewing
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   English Plate Pipeline                 Marathi Plate Pipeline
            │                                         │
   Otsu Binarization & Filter              Devanagari Filtering
            │                                         │
     PyTesseract OCR                       PyTesseract (mar)
            │                                         │
  Pattern Verification               Offline Transliteration (०-९ -> 0-9)
            │                                         │
            └────────────────────┬────────────────────┘
                                 │
                                 ▼
                 Structured Output & CSV Log Entry
```

---

## 🏗️ Architecture

The codebase is structured into a clean modular Python library (`CODE/src/`) separating core computer vision algorithms from web endpoints:

```text
┌─────────────────────────────────────────────────────────────────┐
│                    User Interfaces & APIs                       │
│     Web Dashboard (Flask)   │    REST API Endpoint (/api/v1)    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core ANPR Engine                         │
│                  src/detector.py (YOLOv5 Model)                 │
└─────────────────┬───────────────────────────────┬───────────────┘
                  │                               │
                  ▼                               ▼
┌───────────────────────────────────┐ ┌───────────────────────────┐
│     src/preprocessing.py          │ │     src/ocr_engine.py     │
│ Deskewing, Filtering, Binarization│ │ English & Marathi OCR     │
└───────────────────────────────────┘ └───────────────────────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       src/logger.py                             │
│               Real-time CSV & JSON Audit Trail                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Technology Stack

| Domain | Technology | Purpose |
|---|---|---|
| **Deep Learning** | PyTorch, YOLOv5 | License Plate Detection & Bounding Box Regression |
| **Computer Vision** | OpenCV, NumPy, SciPy | Image Deskewing, Bilateral Filtering, Otsu Thresholding |
| **OCR Engines** | PyTesseract, Tesseract OCR | Optical Character Recognition (English & Marathi) |
| **Web Server & API** | Flask, Gevent WSGI | Interactive Web UI & RESTful JSON Microservice |
| **Data Audit & Logging** | Pandas, JSON | Structured Timestamped Detection Records |

---

## 📊 Model Performance & Results

Evaluated on test dataset captures (front and rear vehicle views under varying lighting conditions):

| Metric | Score | Details |
|---|---|---|
| **Precision** | `0.629` | High specificity across vehicle types |
| **Recall** | `0.943` | Near-complete capture rate of visible license plates |
| **mAP@0.5** | `0.940` | Evaluated at 94% confidence threshold over 100 training epochs |
| **OCR Accuracy** | `91% - 94%` | Cleaned preprocessed ROI input vs raw crop |

---

## 📁 Repository Structure

```text
Automatic-Number-Plate-Recognition/
├── app.py                              # Flask Web Application & REST API Server
├── detect.py                           # CLI Batch Inference Script
├── util.py                             # Image Base64 Encoding Helper
├── requirements.txt                    # Unified Dependency Specification
├── Dockerfile                          # Containerization Configuration
├── README.md                           # Main Portfolio Documentation
│
├── src/                                # Core Custom ANPR Package (Your Solution)
│   ├── __init__.py                     # Package Exports
│   ├── detector.py                     # ANPR Pipeline Controller & YOLOv5 Wrapper
│   ├── preprocessing.py                # Deskewing, Otsu Thresholding & Filtering
│   ├── ocr_engine.py                   # English & Devanagari OCR Engine + Offline Mapper
│   └── logger.py                       # Structured CSV & JSON Audit Logger
│
├── weights/                            # Model Weight Checkpoints
│   └── best.pt                         # Fine-Tuned YOLOv5 License Plate Weights
│
├── yolov5/                             # Isolated YOLOv5 Framework Core
│   ├── models/                         # PyTorch Network Architectures
│   ├── utils/                          # Bounding Box & Datasets Helper Utilities
│   ├── train.py                        # Model Training Script
│   ├── val.py                          # Validation Script
│   └── export.py                       # Model Export Tool
│
├── templates/                          # Web Front-End HTML Views
├── static/                             # Web CSS / JS Assets
├── assets/                             # Test Images & UI Screenshots
│   ├── car.png
│   └── m4.jpg
│
├── paper/                              # Academic Research Publication
│   └── ANPR_Paper_112_ICACC.pdf        # Published IEEE/ICACC Conference Paper
│
└── data/                               # Audit Logs & Database
    └── data.csv                        # Timestamped Detection Log
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.8 - 3.11
- Tesseract OCR engine installed on host OS:
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-mar`
  - **Windows**: Install Tesseract-OCR from installer and ensure `tesseract` is added to PATH.

### 2. Environment Setup

```bash
# Clone repository
git clone https://github.com/PritamMahajan/Automatic-Number-Plate-Recognition.git
cd Automatic-Number-Plate-Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Web Application & REST API

```bash
python app.py
```
Access the interactive web portal at `http://localhost:5000`.

### 4. CLI Batch Inference

Run detection on single images or video streams:

```bash
# Detect English Plate
python detect.py --source assets/car.png --weights weights/best.pt

# Detect Marathi Regional Plate
python detect.py --source assets/m4.jpg --weights weights/best.pt --lang mr
```

### 5. REST API Usage Example

Send a Base64 encoded image to the REST API:

```python
import requests
import base64

with open("car.png", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode("utf-8")

payload = {
    "image_base64": f"data:image/png;base64,{b64_string}",
    "language": "eng"
}

response = requests.post("http://localhost:5000/api/v1/predict", json=payload)
print(response.json())
```

**Response Output:**
```json
{
  "status": "success",
  "detected_plate": "MH20EE7598",
  "confidence": 0.943,
  "language": "eng",
  "timestamp": "Wed Sep  9 11:30:00 2026"
}
```

---

## 📜 Academic Citation & Research

If you use this project or reference our ANPR methodology in your research, please cite our **ICACC** conference paper:

```bibtex
@inproceedings{hajare2021anpr,
  title={Automatic Number Plate Recognition System for Indian Number Plates using Machine Learning Techniques},
  author={Hajare, Gayatri and Kharche, Utkarsh and Mahajan, Pritam and Shinde, Apurva},
  booktitle={Proceedings of the International Conference on Advances in Computing and Communications (ICACC)},
  year={2021},
  organization={Ramrao Adik Institute of Technology, D.Y. Patil Deemed to be University}
}
```

---

## 👤 Author & Academic Background

- **Pritam Sunil Mahajan**  
  - **M.Tech in Artificial Intelligence** — *Indian Institute of Technology (IIT) Ropar*  
  - **B.Tech in Computer Engineering** — *Ramrao Adik Institute of Technology, D.Y. Patil Deemed to be University*  
  - **Specialization**: Computer Vision, Deep Learning, Document AI, Multi-lingual OCR Engine Architectures.

---

## 📄 License

This repository is released under the [MIT License](CODE/LICENSE).
