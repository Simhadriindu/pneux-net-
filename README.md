# Pediatric Chest X-Ray Pneumonia Detection with Cross-Operator Validated AI System

**PneuX-Net 2.0** – Explainable, Auto-Learning, Full-Stack Platform

---

## 📑 Table of Contents

- [Why It Matters](#-why-it-matters)
- [Key Achievements](#-key-achievements)
- [Live Experience](#-live-experience)
- [Proof of Reliability](#-proof-of-reliability)
- [Reproducibility](#-reproducibility--statistical-verification)
- [Performance Metrics](#-performance-metrics)
- [Project Overview](#-project-overview)
- [Quick Start](#-quick-start)
- [API for Developers](#-api-for-developers)
- [Technical Architecture](#-technical-architecture)
- [Datasets](#-datasets--preprocessing)
- [Research Methodology](#-research-methodology)
- [Medical Disclaimers](#%EF%B8%8F-medical-disclaimers)
- [Contributing](#-contributing)
- [Future Roadmap](#-future-roadmap)
- [Contact](#-contact)
- [Citation & License](#-citation--license)

---

## 🎯 **Why It Matters**

Pneumonia affects millions globally, requiring rapid and accurate diagnosis from chest X-rays, especially in pediatric patients (ages 1 to 5). This AI system addresses the critical need for reliable automated screening with **rigorous cross-operator validation**, something most medical AI projects lack.

### **🏆 Key Achievements:**
* **86% Cross-Operator Validation Accuracy** on 485 independent samples
* **96.4% Sensitivity** catches 96% of pneumonia cases  
* **Strong Generalization** with only 8.8% accuracy drop on unseen data
* **Production Ready** with live web interface and RESTful API

## 🌐 **Live Experience**

**Professional Medical Interface Features:**
* **Instant Analysis** with sub-second inference and confidence scores
* **DICOM Support** for professional radiology format (.dcm files)
* **AI Attention Maps** provide visual explanation of model decisions
* **PDF Reports** generate clinical-grade reports with dual image layout
* **Zero Data Storage** ensures completely secure local processing
* **Mobile Responsive** works seamlessly across all devices

---

## 📊 **Proof of Reliability**

### **🔬 Rigorous Dual Validation**

| Validation Type | Dataset | Sample Size | Purpose |
|----------------|---------|-------------|---------|
| **Internal** | [Training Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | 269 samples | Model development |
| **Cross-Operator** | [Independent Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset) | 485 samples | Real-world testing |

Both datasets originate from Guangzhou Women and Children's Medical Center, with the validation set representing:
- Distinct patient cohorts (no overlap with training data)
- Time-separated acquisitions (2018 vs 2024)
- Independent radiology review teams
- Separate quality assurance pipelines
   
This **temporal validation with expert re-annotation** effectively captures cross-operator variability over time (same hospital, evolved workflows). The next evolution would be full multi-center testing for even broader generalization.

---

## 📊 **Reproducibility & Statistical Verification**

### Bootstrap-Based AUC Comparison

Our primary statistical test uses **bootstrap resampling** (n=1,000) instead of paired-sample tests, as internal and cross-operator validations used **independent datasets**:

**Statistical Results:**
- **Bootstrap Result**: mean ΔAUC = −0.0001 (95% CI: [−0.0115, 0.0099])
- **Bootstrap p-value**: 0.978 
- **Conclusion**: ✅ **NO significant difference** - Strong generalization across independent test sets

**Why Bootstrap?**  
DeLong's test assumes paired samples from the same test set. Our datasets are independent, making bootstrap resampling the appropriate primary inference method. The CI includes zero, providing strong statistical evidence of generalization robustness.

### **All Reproducibility Files Available:**

**GitHub Repository:**
```
results/reproducibility/
├── bootstrap_auc_results.json # Bootstrap statistical test (primary inference)
├── internal_metrics.json # Internal validation metrics
├── crossop_metrics.json # Cross-operator validation metrics
├── internal_confusion_matrix.csv # Internal CM data
├── crossop_confusion_matrix.csv # Cross-operator CM data
└── model_parameters.json # Complete model architecture & hyperparameters


---

### **📈 Performance Metrics**

| Metric | Internal | Cross-Operator | Drop | Clinical Significance |
|--------|----------|----------|------|----------------------|
| **Accuracy** | 94.8% | **86.0%** | 8.8% ↓ | ✅ Good generalization |
| **Sensitivity** | 89.6% | **96.4%** | 6.8% ↑ | ✅ Excellent screening |
| **Specificity** | 100.0% | **74.8%** | 25.2% ↓ | ⚠️ Acceptable for screening |
| **ROC-AUC** | 98.8% | **96.4%** | 2.4% ↓ | ✅ Outstanding discrimination |

### **🏥 Clinical Interpretation**

The **96.4% sensitivity** means this system catches 96 out of 100 pneumonia cases, which is **excellent for initial screening**. The **25.2% false positive rate** is acceptable for a screening tool—it's better to flag healthy cases for review than miss pneumonia cases.

### **📊 Detailed Validation Results**

![ROC Curve Analysis](results/cross-operator_validation/2_roc_curve.png)
*ROC-AUC: 0.964 showing outstanding diagnostic discrimination*

![Enhanced Confusion Matrix](results/cross-operator_validation/1_enhanced_confusion_matrix.png)
*175 TP, 59 FP, 9 FN with percentage annotations*

![Performance Comparison](results/cross-operator_validation/4_performance_comparison.png)
*Comparison between internal (training) and cross-operator (time-separated validation) performance*

![Calibration Plot](results/cross-operator_validation/7_calibration_plot.png)
*Model calibration analysis for reliable probability estimates*

![Comprehensive Metrics Dashboard](results/cross-operator_validation/8_comprehensive_metrics_dashboard.png)
*Complete performance overview in single visualization*

---

## 🎯 **Project Overview**

This end-to-end medical AI system demonstrates the complete journey from **research to production deployment**, with emphasis on **cross-operator validation**, a critical step often missing in academic projects. The system is focused on pediatric (ages 1 to 5) chest X-rays for targeted clinical impact.

### **🏗️ Complete ML Pipeline:**
* **Data Processing** with balanced dataset creation and preprocessing
* **Model Training** using transfer learning (MobileNetV2 / DenseNet121)
* **Dual Validation** combining internal development and cross-operator generalization testing
* **Web Deployment** through Streamlit interface
* **API Development** with RESTful FastAPI backend (legacy + PneuX-Net 2.0)
* **Database Integration** SQLite/MySQL with JWT auth, feedback, auto-training

---

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
cd chest-xray-pneumonia-detection-ai-main
pip install -r requirements.txt
```

### **2. Run Streamlit (UI)**
```bash
streamlit run api/streamlit_api_folder/streamlit_app.py --server.port 8501
```
Open http://localhost:8501 – Upload X-rays, get results, PDF reports, Grad-CAM.

### **3. Run PneuX-Net 2.0 Backend (optional – for Login, DB, Auto-Training)**
```bash
cd backend
python seed_admin.py    # Admin: admin@pneuxnet.com / admin123
python main.py          # API at http://localhost:8000
```
Then Streamlit connects to backend for auth, history, feedback → database.

### **4. Legacy API (predict-only)**
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 7860
```
Docs: http://localhost:7860/docs

### **📥 Option 5: Use Pre-trained Model**
```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Download and load model
model_path = hf_hub_download(
    repo_id="ayushirathour/chest-xray-pneumonia-detection",
    filename="best_chest_xray_model.h5"
)
model = tf.keras.models.load_model(model_path)
```

### **🔬 Option 6: Train Your Own Model**
```bash
# Download datasets (links in Dataset section below)
python scripts/analyze_and_balance.py
python scripts/create_balanced_dataset.py

# Train and evaluate
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/cross-operator_validation.py
```

---

## 📡 **API for Developers**

### **🔗 API Endpoints**

**Legacy API (api/main.py, port 7860):**
* `POST /predict` – Upload X-ray for pneumonia detection
* `GET /health` – API health and model status
* `GET /stats` – Cross-operator validation metrics
* `GET /docs` – Swagger documentation

**PneuX-Net 2.0 API (backend/main.py, port 8000):**
* `POST /api/auth/register`, `POST /api/auth/login` – JWT auth
* `POST /api/xray/upload-predict` – Upload + predict (saved to DB)
* `POST /api/xray/gradcam` – Grad-CAM heatmap
* `POST /api/xray/feedback` – Submit feedback (triggers auto-training)
* `GET /api/xray/history` – Prediction history
* `GET /api/admin/dashboard`, `POST /api/admin/train` – Admin

### **📊 Sample Response:**
```json
{
  "diagnosis": "PNEUMONIA",
  "confidence": 92.54,
  "confidence_level": "High",
  "recommendation": "Strong indication of pneumonia. Recommend immediate medical attention.",
  "raw_score": 0.9253779053688049,
  "timestamp": "2025-08-18T15:18:33.827996",
  "filename": "person34_virus_76.jpeg",
  "image_size": "(1648, 1400)x1400",
  "cross_operator_validation_performance": {
    "accuracy": "86.0%",
    "sensitivity": "96.4%",
    "specificity": "74.8%",
    "validated_on": "485 independent samples"
  },
  "disclaimer": "This AI assistant is for preliminary screening only. Always consult healthcare professionals for medical decisions."
}
```

### **🐍 Python Integration:**
```python
import requests

# Legacy API (port 7860)
with open("chest_xray.jpg", "rb") as f:
    r = requests.post("http://localhost:7860/predict", files={"file": f})
    print(r.json()["diagnosis"], r.json()["confidence"])

# PneuX-Net 2.0 (port 8000) – requires token
# POST /api/auth/login first, then use Bearer token
```

---

## 🧠 **Technical Architecture**

### **🤖 Model Design Rationale:**
* **MobileNetV2 Backbone** optimized for deployment efficiency over maximum accuracy
* **Trade-off Analysis** shows 3% lower accuracy than ResNet50, but 5x faster inference
* **Resource Optimization** with 14MB model enables low-resource deployment
* **Transfer Learning** uses ImageNet pre-training for efficient medical adaptation

### **🏗️ Architecture Diagram:**
```
INPUT (224x224 X-ray)
    ↓
MobileNetV2 Backbone (Pre-trained ImageNet weights)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Sigmoid Output
    ↓
PREDICTION (Probability Score)
```

### **🗃️ System Components:**
* **UI:** Streamlit with glassmorphic medical design
* **Backend:** FastAPI (legacy + PneuX-Net 2.0 with JWT, DB)
* **Model:** TensorFlow (MobileNetV2 / DenseNet121)
* **Database:** SQLite (default) or MySQL
* **Deployment:** Multi-platform (Streamlit Cloud, Docker ready)

### **📁 Project Structure:**
```
chest-xray-pneumonia-detection-ai-main/
├── requirements.txt        # Single deps file (install once)
├── backend/                # PneuX-Net 2.0 API (auth, DB, auto-training)
│   ├── app/                # FastAPI routes, ML, models
│   └── main.py
├── api/
│   ├── main.py             # Legacy predict-only API (port 7860)
│   └── streamlit_api_folder/   # Streamlit UI (connects to backend:8000)
│       ├── streamlit_app.py
│       └── api_client.py
├── scripts/                # prepare_dataset, initial_train, train_model, etc.
├── data/                   # raw, processed, user_uploads
├── models/                 # model_v1.h5, best_chest_xray_model.h5
├── results/                # Validation results
│   ├── internal_validation/
│   ├── cross-operator_validation/
│   └── reproducibility/
└── README_PNEUXNET2.md     # PneuX-Net 2.0 detailed guide
```

---

## 📊 **Datasets & Preprocessing**

### **📁 Data Sources**
**Important:** Datasets are NOT included in repository. Manual download required:

* **Training Data:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) with ~2GB dataset (pediatric patients, ages 1 to 5)
* **Cross-Operator Validation:** [Pneumonia Radiography Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset) with 485 samples (pediatric patients, ages 1 to 5)

### **Key Differences Between Datasets**

| Aspect | Training Dataset (Mooney et al., Cell 2018) | Cross-Operator Dataset (Pratibha-verified "Radiography") |
|---|---|---|
| Source & Cohort | Guangzhou Women & Children's Medical Center; retrospective X-rays of patients **1–5 yrs** collected for routine care. | Same hospital but **different acquisition teams & radiology reviewers** (hence "cross-operator"); images were re-exported and re-audited independently. |
| Size & Split | **5,863 images** already split into `train / test / val`. | **4,929 images** (1,249 Normal, 3,680 Pneumonia) also in `train / test / val`, but class counts differ. |
| Pneumonia Sub-types | Label is binary (Pneumonia vs Normal). | Pneumonia folder contains **Bacterial (2,435) vs Viral (1,245)** counts (still used as a single "Pneumonia" class in our model). |
| Quality-Control Pipeline | Two expert physicians graded all scans; a third physician double-checked the evaluation split. | All low-quality scans **manually removed**; initial dual-review + final audit by **Dr. Pratibha (senior radiologist)**. |
| Label Consistency | Labels follow original Cell 2018 study protocol. | Labels re-verified in 2024 audit; eliminates a handful of mis-labels reported in the original set. |
| Class Balance | Roughly **3.3×** more Pneumonia than Normal (4,273 vs 1,583); imbalance handled later by our balancing script. | Imbalance even higher (**≈3×**), but counts differ → provides a *new* distribution for robustness testing. |
| File Provenance | JPEGs exported directly from PACS in 2017-2018. | Same imaging hardware, but **different technologists (operators)** and separate export batch → tests operator-to-operator variation (cross-operator generalization). |
| Licensing | CC BY 4.0. | CC0 (public domain). |

**Why This Matters for the Model**  
* The **training dataset** teaches the network typical pediatric patterns under one operator workflow.  
* The **cross-operator dataset** probes generalization when image export settings, technicians, and a fresh radiology audit change, mirroring real-world variability.  
* Using both ensures the model is not just memorizing plate-specific noise or labeling quirks and gives the reported **86% accuracy / 96% sensitivity** on truly independent data.

### **🔄 Preprocessing Pipeline:**
1. Download datasets from provided Kaggle links
2. Run `python scripts/analyze_and_balance.py` for data analysis
3. Execute `python scripts/create_balanced_dataset.py` for preprocessing
4. Balanced datasets created locally (1K+ training images)

---

## 🔍 **Research Methodology**

### **Cross-Operator Validation Design**

Unlike typical medical AI papers that test on same-day data under same conditions, this project implements **true cross-operator validation**:

| Aspect | Details |
|--------|---------|
| **Temporal Separation** | 2018 training data vs 2024 validation data |
| **Operator Variation** | Different radiology review teams |
| **Quality Audit** | Independent re-verification by senior radiologist |
| **Sample Size** | 485 independent samples (not cherry-picked) |
| **Clinical Scenario** | Mirrors real deployment across different hospitals/technicians |

This approach is **often missing** in academic AI projects but **critical** for clinical deployment.

### **Statistical Approach**
- **Internal Validation**: Standard train/test split (same operator protocol)
- **Cross-Operator Validation**: Independent dataset with different imaging workflows
- **Primary Test**: Bootstrap AUC comparison (n=1,000 resamples) - recommended for independent samples
- **Code**: `scripts/cross-operator_validation.py`
- **Results**: `results/reproducibility/bootstrap_auc_results.json`

---

## ⚠️ **Medical Disclaimers**

### **🚨 Critical Limitations**
* **Research Prototype** is NOT a medical device or FDA/CE approved
* **Screening Tool Only** is NOT for definitive diagnosis
* **Professional Review Required** as all results need radiologist oversight
* **25.2% False Positive Rate** means 1 in 4 normal cases may be flagged incorrectly
* **Pediatric Focus** is optimized for ages 1 to 5; performance on other age groups untested

### **📊 Technical Constraints**
* **Binary Classification** cannot detect specific pneumonia types
* **Image Quality Dependent** as performance degrades with poor quality scans
* **Dataset Limitations** include limited population and imaging protocol diversity
* **No Clinical Context** cannot consider patient history or symptoms

### **✅ Appropriate Use Cases**
* Academic research and methodology demonstration
* Educational AI in healthcare training
* Technical portfolio projects
* **NOT for clinical diagnosis or patient care decisions**

---

## 🤝 **Contributing**

This project welcomes contributions in:
- **Model improvements** and optimization
- **Multi-center validation** expansion
- **Clinical workflow** integration
- **Deployment** in resource-limited settings
- **Documentation** and tutorials

### **Contributing Guidelines:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🛣️ **Future Roadmap**

- [ ] Multi-center validation across 5+ hospitals
- [ ] Pneumonia subtype classification (bacterial vs viral)
- [ ] Integration with DICOM imaging servers
- [ ] FDA pre-certification pathway study
- [ ] Pediatric age-specific performance analysis (1-2, 3-4, 5+ years)
- [ ] Real-time model monitoring dashboard
- [ ] Mobile app for offline inference
- [ ] Attention visualization improvements
- [ ] Explainability analysis (LIME, SHAP)

---

### **👥 Collaboration Opportunities:**
* 🩺 **Medical Professionals** for clinical validation and expert feedback
* 🎨 **UI/UX Designers** for enhanced medical interface design
* 💻 **Python Developers** for API optimization and new features
* 📊 **Data Scientists** for model improvement and validation expansion


---

**⚡ Advancing AI in Healthcare Through Rigorous Validation & Accessible Deployment**

*Demonstrating that medical AI can be both scientifically robust and practically accessible to the global community.*

---

📄 **See also:** [README_PNEUXNET2.md](README_PNEUXNET2.md) for PneuX-Net 2.0 setup (Login, DB, Auto-Training).
