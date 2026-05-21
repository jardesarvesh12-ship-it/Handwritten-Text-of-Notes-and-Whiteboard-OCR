# Handwritten-Text-of-Notes-and-Whiteboard-OCR

# An advanced production-ready OCR system that extracts text from:
- Handwritten notes
- Classroom whiteboards
- Flowcharts
- Printed documents
- Scanned images

# The system automatically handles:
✅ Shadows  
✅ Glare  
✅ Tilt correction  
✅ Noise removal  
✅ Image enhancement

# Features:
- Drag & Drop Upload
- Before/After Image Preview
- Copy Extracted Text
- OCR Engine Selection
- Auto Pipeline Detection

🔌 REST API Support
- Multipart image upload
- JSON API
- Base64 image support

 📊 Evaluation Metrics
Supports OCR benchmarking:
- CER (Character Error Rate)
- WER (Word Error Rate)

 🗄️ Database Support
- SQLite database integration
- Stores extracted OCR text automatically

# 📂 Project Structure
```text
handwritten-ocr/
├── app/
│   ├── __init__.py
│   └── main.py
├── utils/
│   ├── preprocessor.py
│   └── ocr_engine.py
├── models/
│   └── evaluate.py
├── dataset/
│   ├── prepare_dataset.py
│   ├── samples/
│   ├── synthetic/
│   └── dataset_meta.json
├── instance/
│   └── ocr_database.db
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
├── tests/
├── docs/
├── ocr_cli.py
├── requirements.txt
├── .env.example
└── README.md

# ⚙️ Installation Guide

 📌 Prerequisites
- Python 3.9+
- Git
- Tesseract OCR (Optional)

 🔧 Step 1 — Clone Repository
git clone https://github.com/jardesarvesh12-ship-it/Handwritten-Text-of-Notes-and-Whiteboard-OCR.git
cd Handwritten-Text-of-Notes-and-Whiteboard-OCR

 🐍 Step 2 — Create Virtual Environment
 Windows
python -m venv venv
venv\Scripts\activate
cd handwritten-ocr 


 📦 Step 3 — Install Dependencies
pip install -r requirements.txt

 🔤 Step 4 — Install Tesseract OCR (Optional)

Download:
https://github.com/UB-Mannheim/tesseract/wiki

Install to:
C:\Program Files\Tesseract-OCR\

Add to PATH or set:
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe



 ⚙️ Step 5 — Configure Environment
cp .env.example .env
Edit `.env` if needed.


 🧪 Step 6 — Prepare Dataset (Optional)
python dataset/prepare_dataset.py

# ▶️ Step 7 — Run Application
python app/main.py




# System Architecture


        User Image
            │
            ▼
┌─────────────────────────────────┐
│       ImagePreprocessor         │
│  • Auto-detect pipeline         │
│  • Tilt correction              │
│  • Shadow removal               │
│  • CLAHE enhancement            │
│  • Binarization                 │
│  • Denoising                    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│          OCRManager             │
│  ┌──────────┐  ┌─────────────┐  │
│  │ EasyOCR  │  │ Tesseract   │  │
│  └──────────┘  └─────────────┘  │
│        ┌──────────────┐         │
│        │    TrOCR     │         │
│        └──────────────┘         │
│  → Best confidence selected     │
└────────────────┬────────────────┘
                 │
                 ▼
           Extracted Text
                 │
                 ▼
         SQLite Database




















