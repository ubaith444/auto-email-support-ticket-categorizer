# GEMINI.md

# Auto Email / Support Ticket Categorizer
### AI / ML Intern Technical Assessment

---

# Project Overview

You are an expert AI/ML Engineer, NLP Engineer, Software Engineer, Data Scientist, MLOps Engineer, Solution Architect, Technical Writer, and System Designer.

Your task is to design and build a production-quality solution for the following assessment.

---

# Assessment Statement

Build a lightweight NLP classifier that reads an incoming support ticket and routes it to the correct department automatically and in real time.

This project should simulate the intelligent ticket triage layer used in enterprise helpdesk systems.

The system should classify every incoming ticket into one of the following departments:

- BILLING
- TECHNICAL
- HR
- GENERAL

---

# Core Objective

Do NOT build just another ML notebook.

Instead, design the project exactly like a real-world Machine Learning Engineering project.

The repository should demonstrate:

- Software Engineering
- Machine Learning Engineering
- NLP
- System Design
- Data Engineering
- Clean Architecture
- Documentation
- Testing
- Maintainability
- Production Readiness

The final repository should look like something built by an experienced ML Engineer rather than a student assignment.

---

# Development Principles

Follow

- SOLID Principles
- DRY
- KISS
- Separation of Concerns
- Modular Architecture
- Reusable Components
- Configuration Driven Design
- Clean Code
- PEP8
- Type Hints
- Comprehensive Docstrings

---

# Dataset

Use the **IT Support Ticket Dataset** as the primary dataset.

Before implementation:

1. Inspect the dataset completely.
2. Explain every column.
3. Identify the target column.
4. Check missing values.
5. Check duplicates.
6. Check label distribution.
7. Identify data quality issues.

If dataset labels are different from

- BILLING
- TECHNICAL
- HR
- GENERAL

create a documented mapping strategy.

Every mapping decision must be justified.

If necessary, augment the dataset with realistic synthetic samples.

Never fabricate statistics.

Always compute actual values.

---

# Project Lifecycle

Follow an end-to-end SDLC + ML Lifecycle.

Include documentation for every phase.

1. Project Initiation

2. Requirement Analysis

3. Business Understanding

4. Problem Definition

5. Objectives

6. Scope

7. Stakeholders

8. Functional Requirements

9. Non Functional Requirements

10. Project Planning

11. Risk Analysis

12. Technology Stack

13. Dataset Understanding

14. Exploratory Data Analysis

15. Data Cleaning

16. Data Validation

17. Feature Engineering

18. NLP Pipeline

19. Model Selection

20. Hyperparameter Tuning

21. Training

22. Evaluation

23. Error Analysis

24. Model Improvement

25. Prediction Pipeline

26. Deployment Strategy

27. Monitoring

28. Testing

29. Documentation

30. Project Closure

Every section must contain

- explanation
- purpose
- implementation
- decisions
- assumptions
- trade-offs

---

# Business Problem

Incoming customer support tickets arrive continuously.

Reading each ticket manually wastes time.

The goal is to automatically identify which department should receive the ticket.

Departments

Billing Team

↓

Technical Team

↓

HR Team

↓

General Support Team

The system should predict the department immediately after receiving the ticket.

---

# Machine Learning Problem

Type

Supervised Learning

Subtype

Multi-Class Text Classification

Input

Raw support ticket text

Output

Department Label

Classes

- BILLING
- TECHNICAL
- HR
- GENERAL

---

# NLP Pipeline

Design an industrial pipeline.

Incoming Ticket

↓

Cleaning

↓

Normalization

↓

Tokenization

↓

Stopword Removal

↓

Optional Lemmatization

↓

TF-IDF Vectorization

↓

Feature Matrix

↓

Classification Model

↓

Prediction

↓

Department Router

Explain every stage.

---

# Preprocessing

Implement

Lowercase conversion

Whitespace normalization

Remove punctuation

Remove URLs

Remove email addresses

Remove HTML

Remove special characters

Remove duplicate spaces

Handle empty text

Handle null values

Remove duplicates

Optional

Lemmatization

Document why each preprocessing step exists.

---

# Exploratory Data Analysis

Generate

Dataset Shape

Dataset Information

Missing Values

Duplicate Analysis

Class Distribution

Average Ticket Length

Vocabulary Size

Top Words

Department Frequency

Word Clouds (optional)

Ticket Length Distribution

Explain every visualization.

---

# Feature Engineering

Implement

TF-IDF

Experiment with

- Unigrams
- Bigrams
- Trigrams

Tune

max_features

min_df

max_df

Document why final parameters were selected.

---

# Model Selection

Train and compare

Logistic Regression

Multinomial Naive Bayes

Linear SVM

Decision Tree

Random Forest

Evaluate

Accuracy

Precision

Recall

F1 Score

Training Time

Prediction Time

Model Size

Select the best model.

Explain why.

---

# Model Evaluation

Generate

Classification Report

Confusion Matrix

Accuracy

Precision

Recall

F1

Cross Validation Score

Misclassified Samples

Never invent metrics.

Only display actual computed values.

---

# Prediction Module

Support

Single Prediction

Batch Prediction

Interactive CLI

Prediction Confidence

Example

Input

Unable to login after password reset.

Prediction

TECHNICAL

Confidence

98.4%

---

# System Architecture

Design complete architecture.

Include ASCII diagrams.

Example

User

↓

Support Ticket

↓

Prediction API

↓

Preprocessing Layer

↓

TF-IDF

↓

ML Model

↓

Department Router

↓

Department Queue

Also create

- High Level Architecture
- Component Diagram
- Data Flow Diagram
- Sequence Diagram
- ML Pipeline Diagram
- Deployment Diagram

---

# Project Structure

Design enterprise folder structure.

Example

project/

    data/
        raw/
        processed/

    notebooks/

    src/
        preprocessing/
        features/
        models/
        evaluation/
        prediction/
        visualization/
        utils/

    config/

    artifacts/

    reports/

    tests/

    docs/

    models/

    requirements.txt

    README.md

    main.py

---

# Code Standards

Every file should

Use logging

Use type hints

Use docstrings

Handle exceptions

Avoid duplicated logic

Separate business logic

Create reusable modules

Use configuration files

Follow PEP8

---

# Testing

Implement

Unit Tests

Integration Tests

Prediction Tests

Edge Cases

Invalid Input Tests

Empty Input Tests

Unknown Ticket Tests

---

# Documentation

Generate

README

Architecture Documentation

Model Documentation

Installation Guide

Usage Guide

API Documentation (future)

Screenshots Placeholders

Future Enhancements

Business Benefits

Lessons Learned

Limitations

---

# Deployment Strategy

Explain deployment using

FastAPI

Flask

Docker

Render

Railway

AWS

Azure

Do not deploy.

Only design deployment architecture.

---

# Future Enhancements

Discuss

Transformer Models

BERT

DistilBERT

Sentence Transformers

Online Learning

Active Learning

Feedback Loop

Model Monitoring

MLflow

CI/CD

REST API

Authentication

Cloud Deployment

Real-time Queue Processing

Kafka

Redis

---

# Deliverables

Generate

- Complete source code
- Modular architecture
- Clean documentation
- Diagrams
- Training scripts
- Prediction scripts
- Evaluation scripts
- Configuration
- Requirements
- README
- Tests

Everything should be production-ready.

---

# Important Constraints

- Prioritize correctness over unnecessary complexity.
- Keep the core implementation lightweight (TF-IDF + classical ML) to match the assessment requirements.
- Do not introduce LLMs or transformer models into the main solution.
- Ensure all evaluation metrics are computed from actual model results.
- Write clear, maintainable, and well-documented code suitable for review by technical interviewers.
- The final project should satisfy the assessment requirements while also being strong enough to showcase in a professional GitHub portfolio.