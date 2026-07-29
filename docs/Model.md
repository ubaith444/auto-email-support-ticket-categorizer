# Machine Learning Model & Optimization Documentation

## 1. Problem Framing & Dataset Audit
- **Task**: Supervised Multi-Class Text Classification.
- **Classes**: `BILLING`, `TECHNICAL`, `HR`, `GENERAL`.
- **Dataset Size**: **473 samples** (HR: 120, BILLING: 119, TECHNICAL: 119, GENERAL: 115).
- **Target Metric**: Repeated Stratified 5-Fold Cross-Validation F1-Weighted & Held-Out Test Accuracy (95 samples).

---

## 2. Methodology & Leakage Prevention Policy
- **Train/Test Split**: 80% Training (`X_train`, 378 samples), 20% Held-Out Test (`X_test`, 95 samples).
- **Strict Data Leakage Prevention**:
  - `TfidfVectorizer` is encapsulated strictly inside scikit-learn `Pipeline` objects.
  - During `GridSearchCV` and `RepeatedStratifiedKFold` (5 splits x 3 repeats = 15 splits), feature vocabulary and IDF weights are fit **ONLY** on training folds.
  - Evaluation reporting is performed exclusively on the held-out `X_test` dataset.

---

## 3. Candidate Model Benchmark Comparison

All candidate algorithms were optimized via `GridSearchCV` hyperparameter tuning and evaluated across 15 validation folds:

| Algorithm | Optimal Hyperparameters | Repeated CV F1 (Mean ± Std) | Held-Out Test Acc | Fit Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `{'C': 2.0, 'solver': 'lbfgs'}` | **87.45% ± 4.10%** | **91.58%** | **0.922s** | **SELECTED OPTIMAL** |
| **Linear SVM** | `{'C': 1.0}` | 87.05% ± 4.31% | 90.52% | 6.454s | Candidate |
| **Multinomial Naive Bayes** | `{'alpha': 1.0}` | 84.54% ± 3.67% | 88.42% | 0.459s | Candidate |
| **Random Forest** | `{'n_estimators': 200}` | 80.54% ± 4.34% | 82.10% | 10.032s | Candidate |
| **Decision Tree** | `{'max_depth': 20}` | 60.31% ± 6.75% | 63.15% | 0.638s | Candidate |

---

## 4. Feature Importance Insights (Top Words per Class)

### 💳 BILLING
Top Terms: `invoice`, `charged`, `refund`, `payment`, `receipt`, `card`, `subscription`, `billing`, `twice`, `statement`.

### 🛠️ TECHNICAL
Top Terms: `app`, `login`, `error`, `server`, `crashes`, `api`, `password`, `code`, `failing`, `database`.

### 👥 HR
Top Terms: `leave`, `salary`, `attendance`, `hr`, `request`, `letter`, `slip`, `details`, `approval`, `payroll`.

### 💬 GENERAL
Top Terms: `office`, `support`, `hours`, `contact`, `located`, `pricing`, `demo`, `policy`, `customer`, `where`.

---

## 5. Artifact Serialization
- Saved model pipeline: `artifacts/best_model.joblib`
- Metrics & feature importance report: `artifacts/metrics.json`
