# PneuX-Net 2.0 - Setup Guide (2026 B.Tech Project)

## Step-by-Step Setup (Terminal by Terminal)

---

### Step 1: Open Terminal 1 – Project & Install

**Terminal 1** (PowerShell / CMD / Bash):

```bash
cd chest-xray-pneumonia-detection-ai-main
pip install -r requirements.txt
```

*Wait for install to complete.*

---

### Step 2: Download Dataset (Browser)

1. Open: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  
2. Click **Download** (requires Kaggle account)  
3. Unzip the downloaded file  
4. Copy/move contents to:  
   `chest-xray-pneumonia-detection-ai-main/data/raw/chest_xray/`  

**Required structure:**
```
data/raw/chest_xray/
├── train/NORMAL/
├── train/PNEUMONIA/
├── test/NORMAL/
├── test/PNEUMONIA/
├── val/NORMAL/
└── val/PNEUMONIA/
```

---

### Step 3: Terminal 1 – Prepare Dataset

**Terminal 1** (same terminal, project root):

```bash
python scripts/prepare_dataset.py
```

*Creates `data/processed/` and `data/dataset_metadata.csv`.*

---

### Step 4: Terminal 1 – Train Model (Optional)

**Terminal 1** (same terminal):

```bash
python scripts/initial_train.py
```

*Saves `models/model_v1.h5`. Takes ~30–60 min on GPU.*  
*Skip if needed – system runs with fallback (lower accuracy).*

---

### Step 5: Terminal 1 – Start Backend

**Terminal 1** (same terminal):

```bash
cd backend
python seed_admin.py
python main.py
```

*Leave this running. Backend at http://localhost:8000.*  
*Admin: `admin@pneuxnet.com` / `admin123`.*

---

### Step 6: Open Terminal 2 – Start Streamlit

**Open a NEW terminal** (Terminal 2):

```bash
cd chest-xray-pneumonia-detection-ai-main
streamlit run api/streamlit_api_folder/streamlit_app.py --server.port 8501
```

*Leave this running. UI at http://localhost:8501.*

*Or run `run_streamlit.bat` from project root.*

---

### Step 7: Use the System (Browser)

1. Open **http://localhost:8501**  
2. **Login** → `admin@pneuxnet.com` / `admin123` (or **Register**)  
3. **Upload & Analyze** → upload X-ray → **Analyze with AI**  
4. View prediction, Grad-CAM, PDF report  
5. Give feedback (Yes/No) – saved to database  

---

## Quick Reference (Terminal by Terminal)

| Step | Terminal | Command |
|------|----------|---------|
| 1 | Terminal 1 | `cd chest-xray-pneumonia-detection-ai-main` → `pip install -r requirements.txt` |
| 2 | Browser | Download Kaggle dataset → unzip to `data/raw/chest_xray/` |
| 3 | Terminal 1 | `python scripts/prepare_dataset.py` |
| 4 | Terminal 1 | `python scripts/initial_train.py` (optional) |
| 5 | Terminal 1 | `cd backend` → `python seed_admin.py` → `python main.py` *(keep running)* |
| 6 | Terminal 2 | `cd chest-xray-pneumonia-detection-ai-main` → `streamlit run api/streamlit_api_folder/streamlit_app.py --server.port 8501` *(keep running)* |
| 7 | Browser | Open http://localhost:8501 |

---

## Project Mapping

| Component | Path | Port |
|-----------|------|------|
| Root deps | `requirements.txt` | - |
| Backend API | `backend/main.py` | 8000 |
| Streamlit UI | `api/streamlit_api_folder/streamlit_app.py` | 8501 |
| Legacy API | `api/main.py` | 7860 |
| Scripts | `scripts/*.py` | - |
| Data | `data/raw/`, `data/processed/`, `data/user_uploads/` | - |
| Models | `models/*.h5` | - |
| Database | `pneuxnet.db` (SQLite) | - |

---

## Without Pre-trained Model

If you skip Step 5 (no `model_v1.h5`):
- System still runs
- Predictions use fallback model (lower accuracy)
- Run `initial_train.py` when dataset is ready

---

## MySQL (Optional)

Create database `pneuxnet_db`, then add `.env` in project root:

```
USE_MYSQL=true
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=pneuxnet_db
```

---

## Admin Features

| Feature | Where |
|---------|-------|
| Upload Dataset ZIP | Sidebar → Admin → Upload ZIP |
| Trigger Retrain | Admin → Trigger Retrain |
| Training Logs | Admin → Training Logs |
| Auto-Training | ON by default; feedback triggers retrain |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python scripts/initial_train.py` |
| Backend not connecting | Start backend before Streamlit; check port 8000 |
| DB error | Delete `pneuxnet.db`, restart backend |
| Grad-CAM fails | `pip install opencv-python-headless` |
| Import errors | Run `pip install -r requirements.txt` from project root |

---

## Docs

- **README_PNEUXNET2.md** – PneuX-Net 2.0 overview  
- **README.md** – Full project documentation  
- **SETUP_GUIDE.md** – This file  


tep 1: Terminal 1 → cd + pip install
Step 2: Browser → Dataset download
Step 3: Terminal 1 → prepare_dataset.py
Step 4: Terminal 1 → initial_train.py
Step 5: Terminal 1 → Backend (keep running)
Step 6: Terminal 2 → Streamlit (keep running)
Step 7: Browser → http://localhost:8501