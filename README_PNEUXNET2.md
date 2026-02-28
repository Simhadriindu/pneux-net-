# PneuX-Net 2.0

**Explainable, Auto-Learning, Full-Stack Deep Learning Platform for Real-Time Pneumonia Detection from Chest X-Rays**

B.Tech Final Year Project 2026

---

## Features

- **DenseNet121** transfer learning for pneumonia detection (96%+ accuracy target)
- **Grad-CAM** explainable AI (heatmap overlay)
- **Auto-training** from user feedback and uploaded data
- **Full-stack** Streamlit + FastAPI + SQLite/MySQL
- **JWT auth** with Login/Register
- **Admin panel** for dataset upload, retraining, logs
- **Chatbot** (PneuX-Bot) for FAQs

---

## Quick Start

### 1. Dataset

Download from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract to `data/raw/chest_xray/` with structure:
```
chest_xray/
  train/NORMAL/
  train/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
```

### 2. Install Dependencies (once)

```bash
# From project root
cd chest-xray-pneumonia-detection-ai-main
pip install -r requirements.txt
```

### 3. Prepare Dataset & Train

```bash
python scripts/prepare_dataset.py
python scripts/initial_train.py
```

### 4. Backend

```bash
cd backend
python seed_admin.py   # Creates admin: admin@pneuxnet.com / admin123
python main.py         # Runs on http://localhost:8000
```

### 5. Streamlit UI

```bash
# From project root
streamlit run api/streamlit_api_folder/streamlit_app.py --server.port 8501
# Or: run_streamlit.bat
# Runs at http://localhost:8501
```

### 6. First Run

1. Start **Backend** first: `cd backend && python main.py`
2. Start **Streamlit**: `streamlit run api/streamlit_api_folder/streamlit_app.py`
3. Open http://localhost:8501
4. Click **Register** or **Login** (admin@pneuxnet.com / admin123)
5. **Upload & Analyze** → upload X-ray → **Analyze with AI**
6. View prediction, Grad-CAM heatmap, and give feedback (saved to database)

---

## Project Structure & Mapping

```
chest-xray-pneumonia-detection-ai-main/
├── requirements.txt            # Single deps file (backend + streamlit + scripts)
├── backend/                    # PneuX-Net 2.0 API (port 8000)
│   ├── app/
│   │   ├── api/                # Auth, X-ray, Admin routes
│   │   ├── core/               # Config, DB, Security
│   │   ├── ml/                 # DenseNet121, Grad-CAM, Auto-trainer
│   │   ├── models/             # SQLAlchemy models
│   │   └── services/
│   ├── main.py
│   └── requirements.txt        # → uses root requirements.txt
├── api/
│   ├── main.py                 # Legacy (old predict-only API)
│   └── streamlit_api_folder/   # Streamlit UI → connects to backend:8000
│       ├── streamlit_app.py
│       ├── api_client.py       # Calls backend /api/* endpoints
│       └── requirements.txt    # → uses root requirements.txt
├── data/
│   ├── raw/chest_xray/         # Kaggle dataset
│   ├── processed/              # Train/val/test
│   ├── user_uploads/
│   └── dataset_metadata.csv
├── models/                     # model_v1.h5, model_v2.h5, ...
├── scripts/
│   ├── prepare_dataset.py
│   └── initial_train.py
└── run_streamlit.bat           # Quick start Streamlit
```

---

## Streamlit Pages

| Page | Description |
|------|-------------|
| Home | Hero, stats, upload section |
| Upload & Analyze | Upload X-ray, get AI prediction |
| Login / Register | JWT authentication |
| Dashboard | Total scans, pneumonia count, avg confidence |
| History | Prediction history from database |
| Chatbot | PneuX-Bot FAQs |
| Admin | Dataset upload, retrain, training logs (admin role) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | Register |
| POST | /api/auth/login | Login |
| POST | /api/xray/upload-predict | Upload X-ray, get prediction |
| POST | /api/xray/gradcam | Get Grad-CAM heatmap |
| POST | /api/xray/feedback | Submit feedback |
| GET | /api/xray/history | Prediction history |
| GET | /api/admin/dashboard | Admin stats |
| POST | /api/admin/train | Trigger retrain |
| POST | /api/admin/upload-dataset | Upload ZIP dataset |

---

## Admin

- First registered user = admin
- Or run `python backend/seed_admin.py` for admin@pneuxnet.com / admin123
- Admin can: upload dataset ZIP, trigger retrain, view training logs

---

## Database

- **SQLite** (default): `pneuxnet.db` in backend folder
- **MySQL**: Set `USE_MYSQL=true`, `MYSQL_PASSWORD`, etc. in `.env`

---

## Design System

- Primary: #0F4C75 (Medical Blue)
- Accent: #00B4D8 (AI Cyan)
- Success: #2ECC71
- Danger: #E63946
- Background: #F8FAFC

---

## Disclaimer

AI screening only. Always consult healthcare professionals for medical decisions.
