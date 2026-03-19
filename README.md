Memory Page Compressibility Predictor (ML + Systems)

A machine learning-based system to predict memory page compressibility, designed to optimize compression decisions in operating systems such as zram and swap.

---

Overview

Memory compression improves memory utilization but introduces CPU overhead. Compressing already incompressible pages wastes time and computational resources.

This project builds an intelligent predictor using machine learning and system-level insights to determine whether a memory page should be compressed before actual compression, improving efficiency and reducing unnecessary computation.

---

Key Highlights

- High Accuracy: ~97–98% across multiple machine learning models
- Fast Prediction: Lightweight feature-based inference
- Models Used:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- Feature Engineering:
  - Distinct byte count
  - Maximum frequency ratio
  - Zero ratio
  - Run-length patterns
- Interactive Interface:
  - Streamlit-based web application for real-time prediction
- System-Level Integration:
  - C-based implementation for kernel-level usage

---

Project Pipeline

Phase 1 – Data Generation

- Generate synthetic memory pages (".bin" files)
- Label pages as compressible or incompressible

Phase 2 – Feature Extraction and Modeling

- Extract features from memory pages
- Train machine learning models (LR, RF, SVM, XGBoost)
- Evaluate and compare performance

Phase 3 – Optimization

- Threshold tuning
- Model refinement
- Performance improvement

---

Model Performance

Model| Accuracy
Logistic Regression| ~97.7%
Random Forest| ~97.9%
SVM| ~97.6%
XGBoost| ~97.8%

Tree-based models performed best due to their ability to handle structured features effectively.

---

Project Structure

├── kernel_implementation/   # C code for kernel-level integration
├── phase1/                  # Data generation and labeling
├── phase1b/                 # Real-world dataset handling
├── phase2/                  # Feature extraction and prediction
├── phase2b/                 # Model evaluation
├── phase3/                  # Optimization and tuning
├── phase3b/                 # Advanced improvements
├── feature_importance.py    # Feature importance analysis
├── model_comparison.py      # Model benchmarking
├── train_ml_models.py       # Training pipeline
├── streamlit_app.py         # Web application
├── run_all_phases.py        # Complete pipeline execution
├── requirements.txt         # Dependencies
├── *.csv                    # Datasets and results
├── *.png                    # Visualizations

---

Quick Start

# Install dependencies
pip install -r requirements.txt

# Train models
python train_ml_models.py

# Run full pipeline
python run_all_phases.py

# Launch Streamlit application
streamlit run streamlit_app.py

---

Streamlit Application

The interactive dashboard allows users to:

- Upload ".bin" memory page files
- View extracted features
- Obtain predictions:
  - Compressible
  - Incompressible

---

Important Note

Large binary datasets ("pages/" folder containing ".bin" files) are not included in this repository to keep it lightweight.

You can regenerate them using:

python phase1/phase1_generate_pages.py

---

Applications

- Operating system memory optimization (zram, swap)
- Compression-aware system design
- Storage optimization
- Performance-efficient computing

---

Skills Demonstrated

- Machine Learning (Supervised Learning)
- Feature Engineering
- Model Evaluation and Comparison
- Data Analysis and Visualization
- Streamlit Application Development
- Systems and Machine Learning Integration

---

License

MIT License
