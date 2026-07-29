# Machine Learning Pipeline Audit & Data Leakage Prevention Report

## 1. Executive Summary

This report presents a thorough, line-by-line engineering audit of the **Auto Email / Support Ticket Categorizer** machine learning pipeline. 

The primary goal of this audit is to guarantee that every reported evaluation metric accurately reflects true model generalization performance, completely eliminating data leakage, evaluation flaws, or artificial score inflation.

---

## 2. Comprehensive 15-Step Pipeline Audit Matrix

| Audit Step | Pipeline Component | Audit Requirement | Verification Result | Compliance Status |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | **Entire Workflow** | Audit dataset loading, preprocessing, training, CV, evaluation, & inference | All operations follow scikit-learn pipeline encapsulation standards | ✅ **VERIFIED CLEAN** |
| **Step 2** | **Train/Test Split** | Split dataset before model fitting (`test_size=0.2`, `stratify=y`) | `train_test_split` splits data into `df_train` (900 samples) & `df_test` (226 samples) | ✅ **VERIFIED CLEAN** |
| **Step 3** | **Text Preprocessing** | Ensure preprocessing operations do not learn data statistics | Preprocessing consists of stateless string transformations (lowercasing, regex) | ✅ **VERIFIED CLEAN** |
| **Step 4** | **TF-IDF Vectorization** | `TfidfVectorizer.fit()` called strictly on `X_train` | Vectorizer is fit exclusively inside scikit-learn `Pipeline` on `X_train` | ✅ **VERIFIED CLEAN** |
| **Step 5** | **Cross Validation** | Verify TF-IDF vocabulary is fit independently per CV fold | `GridSearchCV` with `RepeatedStratifiedKFold` fits vectorizer per training fold | ✅ **VERIFIED CLEAN** |
| **Step 6** | **Final Evaluation** | Ensure evaluation is performed ONLY on held-out `X_test` | `evaluate_pipeline` receives strictly `df_test` (226 samples) | ✅ **VERIFIED CLEAN** |
| **Step 7** | **Saved Model Artifact** | `joblib.dump()` stores complete pipeline (preprocessing + TF-IDF + classifier) | Model artifact `best_model.joblib` contains the complete end-to-end `Pipeline` | ✅ **VERIFIED CLEAN** |
| **Step 8** | **Prediction Script** | Verify inference code (`predictor.py`) NEVER calls `.fit()` or `.fit_transform()` | `TicketPredictor` only invokes `.predict()` / `.predict_proba()` | ✅ **VERIFIED CLEAN** |
| **Step 9** | **Metrics Calculation** | Ensure metrics use ONLY `y_test` and `X_test` predictions | Accuracy, F1, Precision, Recall, and confusion matrix use `df_test` only | ✅ **VERIFIED CLEAN** |
| **Step 10** | **Visualizations** | Ensure charts use test set predictions only | Confusion matrix & evaluation plots are generated from `df_test` predictions | ✅ **VERIFIED CLEAN** |
| **Step 11** | **Duplicate Leakage** | Detect & drop duplicate tickets prior to train/test split | `preprocess_dataset` executes `.drop_duplicates(subset=['cleaned_ticket'])` before split | ✅ **VERIFIED CLEAN** |
| **Step 12** | **Label Leakage** | Ensure target labels are not present in ticket input features | Input text contains raw customer query without category tags or labels | ✅ **VERIFIED CLEAN** |
| **Step 13** | **Feature Leakage** | Verify no engineered features leak department targets | TF-IDF features are extracted purely from sanitized ticket text | ✅ **VERIFIED CLEAN** |
| **Step 14** | **Experiment Reproduction** | Verify alignment between CV F1 score and held-out test accuracy | CV F1 (97.07%) aligns consistently with Held-out Test Acc (97.35%) | ✅ **VERIFIED CLEAN** |
| **Step 15** | **Audit Documentation** | Document findings, code changes, before vs after, and validation checklist | Documented in this file (`docs/evaluation_audit.md`) | ✅ **VERIFIED CLEAN** |

---

## 3. Detailed Audit Findings & Code Justifications

### A. Preprocessing vs Vectorization Leakage Audit
- **Stateless Preprocessing**: Lowercasing, contraction expansion ("can't" -> "cannot"), HTML tag removal, email/URL stripping, and whitespace normalization do not compute global dataset parameters (mean, variance, vocabulary dictionaries). Therefore, performing string cleaning before splitting is statistically valid.
- **Stateful Feature Extraction**: `TfidfVectorizer` computes document frequencies (`idf`) and builds vocabulary mappings. It **MUST ONLY** fit on `X_train`.
- **Code Enforcement**:
  ```python
  # Pipeline Encapsulation in src/models/train.py
  vectorizer = build_tfidf_vectorizer(config)
  pipeline = Pipeline([
      ("tfidf", vectorizer),
      ("classifier", classifier_instance)
  ])

  # Fits vectorizer ONLY on X_train during training and CV folds
  grid_search = GridSearchCV(pipeline, param_grid=spec["param_grid"], cv=rskf, scoring="f1_weighted")
  grid_search.fit(X_train, y_train)
  ```

---

### B. Held-Out Evaluation Isolation
- **Previous Flaw Identified**: Earlier iterations evaluated `pipeline.predict()` on the full dataset dataframe, causing an artificial 100% evaluation accuracy report.
- **Correction Implemented**: `main.py` explicitly isolates `df_test` (226 samples) for all evaluation reporting and plot generation:
  ```python
  # main.py
  pipeline, df_train, df_test, report, feat_importance = run_training_pipeline(config_path)

  # Compute detailed evaluation STRICTLY on held-out test split (df_test)
  eval_results = evaluate_pipeline(
      pipeline,
      df_test,
      text_col="cleaned_ticket",
      target_col=config["data"]["target_column"],
  )
  ```

---

## 4. Empirical Performance & Generalization Metrics

| Metric | Cross-Validation Score (Training Split) | Held-Out Test Set Score (Test Split) | Generalization Consistency |
| :--- | :--- | :--- | :--- |
| **Sample Count** | **900 tickets** (80% split) | **226 tickets** (20% split) | Independent splits |
| **Accuracy** | **97.07%** | **97.35%** | **+0.28% (Consistent)** |
| **F1-Weighted** | **97.07%** | **97.33%** | **+0.26% (Consistent)** |
| **Misclassified** | N/A | **6 / 226 tickets** | High precision |

---

## 5. Final Validation Checklist

- [x] Dataset loaded once per training workflow.
- [x] `train_test_split` performed with stratification (`stratify=y`) and fixed random seed (`random_state=42`).
- [x] Zero duplicate leakage (`.drop_duplicates()` executed prior to splitting).
- [x] Zero label/feature leakage.
- [x] `TfidfVectorizer` fit strictly on `X_train` via scikit-learn `Pipeline`.
- [x] `TfidfVectorizer` transform strictly on `X_test`.
- [x] `RepeatedStratifiedKFold` cross-validation fits vectorizer independently per training fold.
- [x] Evaluation metrics computed exclusively on `X_test`.
- [x] Saved artifact (`best_model.joblib`) contains the complete end-to-end `Pipeline`.
- [x] Inference engine (`predictor.py`) never invokes `.fit()` or `.fit_transform()`.
