# 🧠 Memory Page Compressibility Predictor (ML-Based)

A machine learning-based system to predict whether a memory page is compressible, helping optimize compression decisions in systems like zram and swap memory.

---

## 🚀 Overview

Memory compression (e.g., zram) improves memory efficiency but introduces CPU overhead. Compressing already incompressible pages wastes time and resources.

This project builds an intelligent ML-based predictor that determines whether a memory page should be compressed **before performing compression**, significantly improving system performance.

---

## 🔥 Key Highlights

* ✅ **High Accuracy**: ~97–98% across multiple ML models
* ⚡ **Fast Prediction**: Lightweight feature-based inference
* 🤖 **Multiple Models Used**:

  * Logistic Regression
  * Random Forest
  * Support Vector Machine (SVM)
  * XGBoost (best performer)
* 📊 **Feature Engineering**:

  * Byte distribution analysis
  * Maximum frequency ratio
  * Zero ratio
  * Run-length patterns
* 🌐 **Interactive UI**:

  * Streamlit app for real-time prediction on `.bin` memory pages
* 🧩 **System-Level Relevance**:

  * Inspired by OS-level memory optimization (zram, swap systems)

---

## 🧠 Machine Learning Pipeline

1. **Data Generation**

   * Synthetic memory pages using structured/random patterns (`phase1`)

2. **Feature Extraction**

   * `feat_distinct`
   * `feat_max_freq_ratio`
   * `feat_zero_ratio`
   * `feat_run_count`

3. **Model Training**

   * Trained multiple classification models
   * Compared performance across models

4. **Evaluation**

   * Accuracy: ~97–98%
   * Balanced precision and recall

5. **Deployment**

   * Streamlit-based web application for predictions

---

## 📊 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~97.7%   |
| Random Forest       | ~97.9%   |
| SVM                 | ~97.6%   |
| XGBoost             | ~97.8%   |

📌 Tree-based models (Random Forest, XGBoost) performed best for structured features.

---

## 🗂️ Project Structure

```
├── phase1/                  # Data generation & labeling
├── phase2/                  # Feature extraction & prediction
├── kernel_implementation/   # C-based system integration
├── train_ml_models.py       # Model training & evaluation
├── feature_importance.py    # Feature importance analysis
├── model_comparison.py      # Model benchmarking
├── streamlit_app.py         # Web application (UI)
├── *.csv                    # Dataset & results
├── *.png                    # Visualizations
```

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train_ml_models.py

# Run Streamlit app
streamlit run streamlit_app.py
```

---

## 🌐 Streamlit Demo

Upload a `.bin` memory page and get:

* Extracted features
* Prediction: **Compressible / Incompressible**

---

## 📌 Important Note

Raw memory page dataset (`.bin` files) is not included in this repository to keep it lightweight.

You can generate data using:

```bash
python phase1/phase1_generate_pages.py
```

---

## 🎯 Applications

* OS Memory Optimization (zram, swap systems)
* Compression-aware system design
* Storage & caching optimization
* Performance-aware computing

---

## 💡 Skills Demonstrated

* Machine Learning (Supervised Learning)
* Feature Engineering
* Model Comparison & Evaluation
* Data Processing & Analysis
* System-Level Thinking (OS + ML integration)
* Streamlit Deployment

---

## 📜 License

MIT License
