# Memory Page Compressibility Predictor (ML + Systems)

A machine learning-based system to predict memory page compressibility, designed to optimize compression decisions in operating systems (e.g., zram, swap).

---

## Overview

Memory compression improves memory utilization but introduces CPU overhead. Compressing already incompressible pages wastes time and resources.

This project builds an intelligent predictor using **machine learning + system-level insights** to determine whether a memory page should be compressed **before actual compression**, improving efficiency and reducing unnecessary computation.

---
## Live Demo

A fully functional version of the project is deployed online:

🔗 https://bit.ly/41hBoaY

**Features:**
- Upload `.bin` memory page files  
- Real-time compressibility prediction  
- Multi-model comparison (RF, XGBoost, SVM)  
- Feature analysis and insights

---

## Key Highlights

* **High Accuracy**: ~97–98% across multiple ML models
* **Fast Prediction**: Lightweight feature-based inference
* **Models Used**:

  * Logistic Regression
  * Random Forest
  * Support Vector Machine (SVM)
  * XGBoost
* **Feature Engineering**:

  * Distinct byte count
  * Maximum frequency ratio
  * Zero ratio
  * Run-length patterns
* **Interactive UI**:

  * Streamlit app for real-time prediction
* **System-Level Integration**:

  * C-based implementation for kernel-level usage

---

## Project Pipeline

### Phase 1 – Data Generation

* Generate synthetic memory pages (`.bin` files)
* Label pages as compressible/incompressible

### Phase 2 – Feature Extraction & ML Modeling

* Extract features from memory pages
* Train ML models (LR, RF, SVM, XGBoost)
* Evaluate and compare performance

### Phase 3 – Optimization

* Threshold tuning
* Model refinement
* Performance improvement

---

## Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~97.7%   |
| Random Forest       | ~97.9%   |
| SVM                 | ~97.6%   |
| XGBoost             | ~97.8%   |

-Tree-based models performed best due to structured feature handling.

---

## Project Structure

```
├── kernel_implementation/   # C code for kernel-level integration
├── phase1/                  # Data generation & labeling
├── phase1b/                 # Real-world dataset handling
├── phase2/                  # Feature extraction & ML prediction
├── phase2b/                 # Model evaluation
├── phase3/                  # Optimization & tuning
├── phase3b/                 # Advanced improvements
├── feature_importance.py    # Feature importance analysis
├── model_comparison.py      # Model benchmarking
├── train_ml_models.py       # Training pipeline
├── streamlit_app.py         # Web app for predictions
├── run_all_phases.py        # Complete pipeline execution
├── requirements.txt         # Dependencies
├── *.csv                    # Datasets & results
├── *.png                    # Visualizations
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train_ml_models.py

# Run full pipeline
python run_all_phases.py

# Launch Streamlit app
streamlit run streamlit_app.py
```

---

## Streamlit App

The interactive dashboard allows users to:

* Upload `.bin` memory page files
* View extracted features
* Get prediction:

  * Compressible
  * Incompressible

---

## Important Note

Large binary datasets (`pages/` folder with `.bin` files) are not included in this repository to keep it lightweight.

You can regenerate them using:

```bash
python phase1/phase1_generate_pages.py
```

---
Important Note

Raw memory page files (".bin") are not included in this repository to keep it lightweight and manageable. These files can be generated using the provided scripts.

python phase1/phase1_generate_pages.py

---

## Applications

* Operating System Memory Optimization (zram, swap)
* Compression-aware system design
* Storage optimization
* Performance-efficient computing

---

## Skills Demonstrated

* Machine Learning (Supervised Learning)
* Feature Engineering
* Model Comparison & Evaluation
* Data Analysis & Visualization
* Streamlit Deployment
* Systems + ML Integration

---

## License

MIT
