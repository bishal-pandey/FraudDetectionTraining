# Fraud Detection Training Pipeline

A machine learning pipeline for training fraud detection models using XGBoost. This project implements an end-to-end training pipeline with MLflow tracking and DVC for data version control.

## 🎯 Overview

This repository contains a complete training pipeline for fraud detection using XGBoost classifier. The pipeline is designed to handle imbalanced datasets, perform feature engineering, train models, and track experiments using MLflow.

## ✨ Features

- **XGBoost Classification**: Utilizes gradient boosting for robust fraud detection
- **Experiment Tracking**: MLflow integration for tracking model performance and parameters
- **Data Version Control**: DVC for managing training datasets
- **Containerized Deployment**: Docker support for reproducible environments
- **Scalable Pipeline**: Modular architecture for easy customization and extension

## 📁 Project Structure

```
FraudDetectionTraining/
│
├── .dvc/                       # DVC configuration
├── models/                     # Trained model artifacts
├── src/                        # Source code
│   ├── training_pipeline.py   # Main training pipeline script
│   └── ...                     # Additional modules
│
├── Data.dvc                    # DVC tracked data file
├── Dockerfile                  # Docker configuration
├── .dockerignore              # Docker ignore file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

## 🛠️ Technologies

- **Python 3.10**: Core programming language
- **XGBoost**: Primary machine learning algorithm
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control
- **scikit-learn**: Data preprocessing and evaluation metrics
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **Docker**: Containerization

## 🤖 Model Training

The training pipeline implements:

- **Data Preprocessing**: Handling missing values, feature scaling, and encoding
- **Class Imbalance Handling**: Techniques to address imbalanced fraud datasets
- **Hyperparameter Tuning**: Optimization of XGBoost parameters
- **Model Evaluation**: Comprehensive metrics including precision, recall, F1-score, and AUC-ROC
- **Model Versioning**: Automatic versioning through MLflow

### Key Training Parameters

Default XGBoost parameters can be configured in the training script:
- Learning rate
- Max depth
- Number of estimators
- Scale pos weight (for imbalanced data)


---

⭐ If you find this project useful, please consider giving it a star!
