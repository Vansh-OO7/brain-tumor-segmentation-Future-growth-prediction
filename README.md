# Automated Brain Tumor Segmentation and Future Progression Forecasting

> AI-powered medical imaging project for **brain tumor segmentation** from multi-modal MRI scans and **future tumor progression forecasting** using machine learning.

## Project Status
✅ Completed (Student Project / Portfolio / GitHub Ready)

---

#  Overview

This project presents an end-to-end intelligent healthcare pipeline that:

1. Accepts **multi-modal MRI scans** (`.h5` format)
2. Detects and segments tumor regions using a deep learning model
3. Estimates current tumor area
4. Predicts future tumor progression
5. Generates a visual clinical-style report
6. Provides an interactive web interface using Streamlit

The system demonstrates how AI can assist in medical image analysis and predictive healthcare workflows.

---

# Objectives

- Segment tumor regions from MRI scans using Deep Learning
- Estimate current tumor size from segmentation mask
- Forecast short-term, mid-term, and long-term tumor progression
- Display clinical insights in an interactive dashboard
- Create a deployable healthcare AI prototype

---

# Dataset Used

## 1. Brain Tumor Segmentation Dataset
- **BraTS 2020 Dataset**
- Multi-modal MRI data
- Used for segmentation model training

## 2. Growth Prediction Dataset
Two-stage data preparation:

### Real Reference Data
- 161 real tumor progression samples used for baseline understanding

### Synthetic Growth Dataset
Generated using predicted tumor areas from segmented samples.

- Tumor-only extracted samples: **7915**
- Used for training progression forecasting model

---

# ⚙️ Complete Pipeline

## Step 1 — Input MRI Scan
User uploads a `.h5` MRI file containing 4 modalities.

## Step 2 — Preprocessing
- Resize image
- Normalize intensities
- Prepare tensor for model input

## Step 3 — Tumor Segmentation
A trained **U-Net** model predicts tumor mask.

## Step 4 — Tumor Area Estimation
Mask pixels are counted and converted into area (cm²).

## Step 5 — Future Progression Forecasting
A regression model predicts:

- Short Term Growth
- Mid Term Growth
- Long Term Growth

## Step 6 — Clinical Report Generation
System returns:

- Segmented tumor image
- Current area
- Forecast values
- Growth percentages
- Progression status

## Step 7 — Web Dashboard
Results shown through premium Streamlit interface.

---

# Models Used

## 🔹 Segmentation Model
### U-Net (TensorFlow / Keras)

Used for semantic segmentation of tumor region from MRI scans.

### Why U-Net?
- Excellent for biomedical segmentation
- Pixel-level accuracy
- Works well on limited medical data

---

## 🔹 Forecasting Model
### Random Forest Regressor (Multi-output)

Used to predict future tumor size.

### Outputs:
- Short Term
- Mid Term
- Long Term

---

# 📊 Sample Model Performance

## Future Growth Prediction Model

| Metric | Value |
|------|------|
| MAE | 3.5277 |
| RMSE | 6.0007 |
| R² Score | 0.8669 |

### Individual Forecast Scores

| Horizon | R² |
|--------|------|
| Short Term | 0.9653 |
| Mid Term | 0.8765 |
| Long Term | 0.7589 |

---

#  Application Features

✅ Upload `.h5` MRI file  
✅ Tumor segmentation visualization  
✅ Tumor area estimation  
✅ Growth forecasting  
✅ Progression status detection  
✅ Clinical style output image  
✅ Premium Streamlit UI  
✅ End-to-end automated pipeline

---

# Project Structure

```bash
brain-tumor-segmentation-future-growth-prediction/
│── streamlit_app.py
│── requirements.txt
│── README.md
│
├── models/
│   ├── segmentation_model_30k.h5
│   └── future_growth_model.pkl
│
├── src/
│   └── analyze_input.py
│
├── uploads/
├── outputs/
└── frontend/
