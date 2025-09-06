# 🛡️ Fraud Detection in Imbalanced Datasets  

## 📌 Project Overview  
This project focuses on detecting fraudulent transactions in a **highly imbalanced dataset**:  
- **Legitimate transactions:** 284,315  
- **Fraudulent transactions:** 492  

The goal was to build an effective model that balances **high recall** (catching fraud) with **high precision** (avoiding too many false alarms).  

---

## 📂 Dataset  
- **Source:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Description:**  
  - Contains anonymized transaction features (`V1`–`V28`), plus `Amount` and `Time`.  
  - Target variable: `Class` (0 = Legitimate, 1 = Fraud).  
  - Severe class imbalance (≈0.17% fraud cases).  

---

## 🔎 Methodology  

### 1. Data Preparation  
- Verified data cleanliness (no nulls).  
- Applied **RobustScaler** to `Amount` and `Time` to minimize outlier influence (11.2% outliers in `Amount`).  

### 2. Modeling Strategies  
- **Strategy 1: Baseline Sampling**  
  - Balanced dataset using **SMOTE** (oversampling frauds to 2,500) + **RandomUnderSampler** (legit transactions down to 2,500).  
  - Trained **XGBoost** on resampled data.  

- **Strategy 2: Weighted XGBoost (Best)**  
  - Trained on **full imbalanced dataset**.  
  - Used `scale_pos_weight` to penalize misclassification of fraud class.  
  - Avoided synthetic data → more robust to real-world deployment.  

- **Strategy 3: Anomaly Detection**  
  - Applied **Isolation Forest** to treat fraud as outliers.  
  - Explored unsupervised detection as a baseline.  

### 3. Optimization  
- Used **Optuna** (100 trials) for hyperparameter tuning of Weighted XGBoost.  
- Evaluated models with **AUPRC (Area Under Precision–Recall Curve)** → best suited for imbalanced classification.  

---

## 📊 Results  

| Strategy              | Precision | Recall | AUPRC  | Notes |
|------------------------|-----------|--------|--------|-------|
| Baseline Sampling      | 0.09      | 0.91   | Low    | Recall high but precision unacceptable (too many false positives). |
| Weighted XGBoost       | 0.87      | 0.86   | 0.861  | Best balance, strong candidate for deployment. |
| Isolation Forest       | 0.34      | 0.31   | Low    | Weak compared to supervised methods. |
| Weighted XGBoost + Optuna | **0.88** | **0.87** | **0.877** | Top performance after hyperparameter tuning. |

---

## ✅ Key Takeaways  
- **SMOTE + Undersampling** → boosted recall but failed in precision.  
- **Weighted XGBoost** → best approach for production, achieving strong balance of recall & precision.  
- **Isolation Forest** → unsuitable as a standalone fraud detection tool.  
- **Optuna tuning** → provided incremental improvement, pushing AUPRC to 0.877.  
- **Ensemble analysis** → no benefit from combining XGBoost and Isolation Forest.  

---

## 🚀 How to Run  

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yaman-77/fraud-detection.git
   cd fraud-detection

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   

3. Run the notebook
   ```bash
   jupyter notebook "Fraud Detection.ipynb"


## 📌 Future Work

Experiment with LightGBM and CatBoost for faster training and potential gains.

Explore deep learning (autoencoders, LSTMs) for anomaly detection.

Deploy a real-time fraud detection API using FastAPI or Streamlit.

Incorporate cost-sensitive evaluation (business impact of fraud vs false alarms).

## 🛠️ Tech Stack

Python (pandas, numpy, matplotlib, seaborn)

Scikit-learn (metrics, preprocessing, model selection)

Imbalanced-learn (SMOTE, RandomUnderSampler)

XGBoost (classification)

Optuna (hyperparameter tuning)

Jupyter Notebook (workflow & documentation)

