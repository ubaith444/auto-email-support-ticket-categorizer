# AI/ML Technical Assessment Report - Auto Email / Support Ticket Categorizer

## 1. Executive Summary

This report documents the systematic optimization, dataset expansion, and empirical evaluation of an **Auto Email / Support Ticket Categorizer**. The solution is engineered as a lightweight, production-grade NLP classification engine that automatically categorizes support tickets into `BILLING`, `TECHNICAL`, `HR`, or `GENERAL` departments in real time.

---

## 2. Methodology & Optimization Summary

1. **Dataset Expansion**: Dataset expanded from 160 to **473 samples** (HR: 120, BILLING: 119, TECHNICAL: 119, GENERAL: 115) with diverse phrasing, contractions, enterprise support tickets, and short/long forms.
2. **Text Preprocessing**: Enhanced cleaner with contraction expansion ("can't" -> "cannot", "don't" -> "do not"), HTML/URL/email stripping, and whitespace normalization.
3. **Hyperparameter Grid Search**: Tuned estimators using `GridSearchCV` on scikit-learn `Pipeline` objects.
4. **Repeated Stratified K-Fold CV**: Evaluated stability across 15 splits (`RepeatedStratifiedKFold(n_splits=5, n_repeats=3)`).
5. **Held-Out Evaluation**: Final metrics computed on 20% held-out test split (`X_test`, 95 samples).

---

## 3. Empirical Benchmark Comparison

| Model | Repeated CV F1 (Mean ± Std) | Held-Out Test Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (`C=2.0`)** | **87.45% ± 4.10%** | **91.58%** | **SELECTED OPTIMAL** |
| Linear SVM (`C=1.0`) | 87.05% ± 4.31% | 90.52% | Candidate |
| Multinomial Naive Bayes (`alpha=1.0`) | 84.54% ± 3.67% | 88.42% | Candidate |
| Random Forest (`n_estimators=200`) | 80.54% ± 4.34% | 82.10% | Candidate |
| Decision Tree (`max_depth=20`) | 60.31% ± 6.75% | 63.15% | Candidate |

---

## 4. Verification & Automated Test Suite

Automated testing was conducted via `python -m pytest tests/ -v`:
- **19 Test Cases Passed** across `test_preprocessing.py`, `test_vectorizer.py`, `test_models.py`, `test_predictor.py`, and `test_edge_cases.py`.

---

## 5. Production Readiness & Deployment Design

The architecture is prepared for containerized deployment:
- **FastAPI / Flask REST Endpoint Integration**: Simple wrapper using `TicketPredictor`.
- **Dockerization**: Standard Python 3.11 slim base image.
- **Cloud Hosting**: Compatible with Render, Railway, AWS ECS, or Azure App Service.
