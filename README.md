# 🏦 End-to-End Loan Default Prediction & Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Pipeline-Production--Ready-brightgreen.svg)]()

A modular, production-ready Machine Learning pipeline engineered to assess and predict borrower loan default risks across 148,670 applications. This project covers dynamic feature segregation, financial skewness correction, target leakage detection and resolution, automated preprocessing with Scikit-Learn's `ColumnTransformer`, and model serialization.

---

## 📌 Problem Statement & Overview

In retail and mortgage banking, identifying credit default risks prior to loan approval is critical to preventing Non-Performing Assets (NPAs) while ensuring viable applicants receive fair approval terms.

* **Dataset Size:** 148,670 rows × 34 raw attributes
* **Target Feature:** `Status` (`0` = Repaid / Non-defaulter, `1` = Default)
* **Class Ratio:** ~75.3% Repaid vs. ~24.7% Defaulted (Moderately Imbalanced)
* **Objective:** Build a robust, leakage-free automated preprocessing pipeline and classification model that predicts default probability solely using information available at application time.

---

## 🚨 Critical Engineering Insight: The Data Leakage Audit

### 1. The Anomaly (Baseline Model AUC = 1.00)
Initial baseline model training across all raw numerical attributes produced an unrealistic **ROC-AUC of 1.00** and perfect classification metrics across both classes.

### 2. Root Cause Analysis (Feature Importance Audit)
Auditing tree split weights revealed that over **68%** of the model's decisions were driven by three specific variables:
* `Interest_rate_spread` (Importance: ~0.32)
* `Upfront_charges` (Importance: ~0.20)
* `rate_of_interest` (Importance: ~0.17)

### 3. Domain Logic
In banking workflows, these metrics represent **post-approval terms**:
* If an application is rejected or defaults early in underwriting, upfront charges and interest rate spreads are never computed (recorded as `NaN`).
* The model learned an operational shortcut: *missing post-approval financial charges = default*.

### 4. Remediation
All post-approval features were systematically pruned from the candidate feature pool, restricting the feature space strictly to pre-decision applicant and property data (`dtir1`, `Credit_Score`, `LTV`, `income`, `property_value`, etc.).

---

## 📊 Final Validated Results (Leakage-Free Model)

* **Algorithm:** Random Forest Classifier (`n_estimators=100`, `max_depth=12`, `random_state=42`)
* **Realistic ROC-AUC Score:** **`0.886`** (High discriminatory power)

```text
Classification Report:
               precision    recall  f1-score   support

           0       0.87      0.99      0.93     22406
           1       0.97      0.55      0.70      7328

    accuracy                           0.88     29734
   macro avg       0.92      0.77      0.81     29734
weighted avg       0.89      0.88      0.87     29734
