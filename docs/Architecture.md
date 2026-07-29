# System Architecture Documentation

## 1. High-Level Architecture Diagram

```
+------------------+         +--------------------------+         +-------------------------+
|                  |         |                          |         |                         |
|  Incoming Ticket | ======> |  Preprocessing & Cleaning | ======> |   TF-IDF Vectorizer     |
|   (Web/Email)    |         |        (Regex)           |         |   (Unigram + Bigram)    |
|                  |         |                          |         |                         |
+------------------+         +--------------------------+         +-------------------------+
                                                                               |
                                                                               v
+------------------+         +--------------------------+         +-------------------------+
|                  |         |                          |         |                         |
| Department Queue | <====== |    Department Router     | <====== |  Calibrated Linear SVM  |
|  (Routing Layer) |         | (Confidence Calculation) |         |   Classification Model  |
|                  |         |                          |         |                         |
+------------------+         +--------------------------+         +-------------------------+
```

---

## 2. Low-Level Component Architecture Diagram

```
+-----------------------------------------------------------------------------------------------+
|                                      CLI Application Layer                                    |
|                        (main.py / train.py / predict.py / interactive)                         |
+-----------------------------------------------------------------------------------------------+
                                                |
             +----------------------------------+----------------------------------+
             |                                                                     |
             v                                                                     v
+--------------------------+                                           +------------------------+
|    Training Pipeline     |                                           |   Prediction Engine    |
|   (src/models/train.py)  |                                           |(src/pred/predictor.py) |
+--------------------------+                                           +------------------------+
   |           |           |                                               |                |
   v           v           v                                               v                v
+------+   +-------+   +-------+                                     +----------+     +-----------+
|Text  |   |TF-IDF |   |Model  |                                     |Text      |     |Pipeline   |
|Cleaner|   |Builder|   |Evaluator                                    |Cleaner   |     |Artifact   |
+------+   +-------+   +-------+                                     +----------+     +-----------+
```

---

## 3. Data Flow Diagram (DFD)

```
[Raw Ticket Text] ---> (1. Sanitize & Normalize Text) ---> [Clean Tokens]
                                                                |
                                                                v
                                                   (2. Extract TF-IDF Matrix)
                                                                |
                                                                v
                                                   [N-Gram Feature Matrix]
                                                                |
                                                                v
                                                   (3. Predict Probabilities)
                                                                |
                                                                v
                                                   [Class Probabilities]
                                                                |
                                                                v
                                                   (4. Department Routing)
                                                                |
                                                                v
                                                   [Assigned Department & Confidence]
```

---

## 4. Sequence Diagram (Real-Time Ticket Routing)

```
User / Helpdesk              Prediction Router             Text Cleaner           ML Pipeline Artifact
      |                             |                           |                         |
      |--- 1. Submit Ticket ------->|                           |                         |
      |                             |--- 2. Clean Text -------->|                         |
      |                             |<-- 3. Return Clean Str ---|                         |
      |                             |                                                     |
      |                             |--- 4. Transform & Predict Probabilities ----------->|
      |                             |<-- 5. Return Prob Array & Top Classes ---------------|
      |                             |
      |--- 6. Return Route -------->|
```

---

## 5. Machine Learning & Prediction Pipeline Diagram

```
+----------------------------------------------------------------------------------------+
|                                    TRAINING PIPELINE                                   |
| Raw CSV -> Preprocessing -> Stratified Train/Test Split -> 5-Fold CV -> Joblib Export  |
+----------------------------------------------------------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------------+
|                                   PREDICTION PIPELINE                                  |
| Input Text -> Lowercase/Regex -> TF-IDF Transform -> Calibrated Prediction -> JSON Res |
+----------------------------------------------------------------------------------------+
```

---

## 6. Module Dependency Diagram

```
main.py / train.py / predict.py
   |
   +---> src.utils.io_utils ---> config/config.yaml
   +---> src.preprocessing.text_cleaner
   +---> src.features.tfidf_builder
   +---> src.models.train ---> src.models.evaluator
   +---> src.prediction.predictor
   +---> src.visualization.plots
```

---

## 7. Deployment Architecture Design (FastAPI + Docker + Cloud)

```
[Incoming Support Request]
            |
            v
   [API Gateway / NGINX]
            |
            v
   [FastAPI Container (Gunicorn + Uvicorn Workers)]
            |
            v
   [Joblib Model Pipeline Artifact loaded in memory]
            |
            v
   [Output JSON -> Redis Queue / Department Webhook]
```
