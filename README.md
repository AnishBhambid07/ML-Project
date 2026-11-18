# Enterprise Employee Attrition Prediction & Workforce Risk Intelligence Platform
### Machine Learning CA1 & CA2 Solo Project Submission

A complete, enterprise-grade predictive HR analytics and workforce retention platform. Trained on real-world benchmark data (1,470 records from the IBM HR Analytics benchmark) and integrated with an interactive Google Form survey protocol.

---

## 🌟 Key Highlights & Academic CA1/CA2 Compliance

1. **Algorithms Employed (Strictly Beyond Simple Linear Models):**
   - **Gradient Boosting Classifier (GBM)** (Production Primary: 85.37% Accuracy, 0.8147 ROC-AUC)
   - **Random Forest Classifier (Bagging Ensemble)** (84.01% Accuracy, 0.8030 ROC-AUC)
   - **Multi-Layer Perceptron (MLP Neural Network)** (86.05% Accuracy, 0.8528 Weighted F1)
   - **Support Vector Machine (SVM with RBF Kernel)** (80.61% Accuracy, 55.32% Recall)
   - **K-Means Clustering (Unsupervised Segmentation)** (K=3 Behavioral Risk Tiers)

2. **Real-World Empirical Dataset:**
   - 1,470 observations, 35 attributes (cleaned and formatted).
   - Real-world imbalanced class distribution (16.1% positive attrition rate).
   - Includes full Google Form manual survey question schema (`google_form_survey_questions.md`) with built-in web collector modal.

3. **100% Offline & Self-Contained Web Interface:**
   - Zero-dependency client-side execution by double-clicking `index.html`.
   - Optional local Python HTTP / REST API server (`backend_server.py`) with `/api/predict` and `/api/metrics` endpoints.
   - Built to strictly match the UI, colors, layout, and graph structures shown in the actuarial reference video.

4. **Complete Deliverables Included:**
   - Ready-to-grade academic report (`CA1_CA2_Project_Report.md`).
   - Full Jupyter Notebook (`notebooks/ML_CA1_CA2_Employee_Attrition.ipynb`).
   - Trained model weights (`models/*.joblib`, `models/model_metrics.json`).
   - Single standalone ZIP archive (`Employee_Attrition_ML_CA1_CA2_Project.zip`).

---

## 📁 Project Directory Structure

```text
employee_attrition_project/
├── index.html                           # Main web platform interface
├── backend_server.py                    # Standalone Python HTTP & REST API server
├── train_models.py                      # Complete end-to-end ML model training pipeline
├── requirements.txt                     # Python library dependencies
├── README.md                            # Project overview & running instructions
├── CA1_CA2_Project_Report.md            # Detailed academic project submission report
├── google_form_survey_questions.md      # Google Form survey schema for manual data collection
│
├── data/
│   ├── employee_attrition.csv           # Cleaned 1,470 benchmark records
│   └── employee_attrition_raw.csv       # Original IBM benchmark dataset
│
├── models/
│   ├── gradient_boosting.joblib         # Serialized Gradient Boosting model
│   ├── random_forest.joblib             # Serialized Random Forest model
│   ├── mlp_classifier.joblib            # Serialized Multi-Layer Perceptron model
│   ├── svm_classifier.joblib            # Serialized Support Vector Machine model
│   ├── kmeans_model.joblib              # Serialized K-Means clustering model
│   ├── scaler.joblib                    # Serialized feature StandardScaler
│   └── model_metrics.json               # Precomputed evaluation metrics & ROC coordinates
│
├── static/
│   ├── css/
│   │   └── styles.css                   # Enterprise SaaS actuarial design system
│   └── js/
│       ├── model_data.js                # Precomputed model weights for 100% offline inference
│       └── app.js                       # Interactive client-side ML engine & SVG visualizer
│
└── notebooks/
    ├── ML_CA1_CA2_Employee_Attrition.ipynb # Fully documented academic Jupyter Notebook
    └── generate_notebook.py             # Notebook generator script
```

---

## 🚀 How to Run the Project

### Option A: Instant Offline Web App (No Python Setup Needed)
1. Navigate to the project directory: `employee_attrition_project/`.
2. Double-click `index.html` in your file explorer, or open it in any modern web browser (Chrome, Edge, Firefox, Safari).
3. The interface will open instantly with 100% functionality: real-time inference, interactive sliders, SVG ROC curves, scatter plots, tipping-point matrix, and portfolio stress simulator.

---

### Option B: Local Python Web & REST API Server
1. Open a terminal or PowerShell in the `employee_attrition_project` directory.
2. (Optional) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the local backend server:
   ```bash
   python backend_server.py
   ```
4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

---

### Option C: Retraining the Machine Learning Models
To execute the training pipeline and regenerate all models and metrics from scratch:
```bash
python train_models.py
```
This script will:
- Partition the 1,470 records into 80% train and 20% test sets (stratified).
- Train Gradient Boosting, Random Forest, MLP, SVM, and K-Means.
- Run 5-fold cross validation.
- Output metrics to the terminal and export serialized `.joblib` and `model_metrics.json` files.

---

### Option D: Running the Jupyter Notebook
Open the notebook in VS Code, Jupyter Lab, or Google Colab:
```bash
jupyter notebook notebooks/ML_CA1_CA2_Employee_Attrition.ipynb
```

---

## 📊 Summary of Model Performance (Test Set Evaluation)

| Metric | Gradient Boosting | Random Forest | MLP Neural Net | Support Vector Machine |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **85.37%** | 84.01% | **86.05%** | 80.61% |
| **ROC-AUC** | **0.8147** | 0.8030 | 0.7788 | **0.8196** |
| **Weighted F1** | **0.8357** | 0.8380 | **0.8528** | 0.8164 |
| **Precision** | **57.7%** | 50.0% | 56.8% | 43.3% |
| **Recall (Sensitivity)**| 31.91% | 46.81% | 44.68% | **55.32%** |
| **5-Fold CV AUC** | **0.818 ± 0.035** | 0.804 ± 0.041 | 0.782 ± 0.038 | **0.820 ± 0.029** |

*Unsupervised K-Means Clustering (K=3): Silhouette Coefficient = 0.0927, Davies-Bouldin = 2.5888.*

---

## 👥 Student & Course Info
- **Project:** Machine Learning CA1 & CA2 Continuous Assessment
- **Role:** Senior Machine Learning Engineer / Lead Data Scientist
- **Submission Type:** Solo Project
- **Status:** Complete, Self-Contained, and Verified
# ML-Project
