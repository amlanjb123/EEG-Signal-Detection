project name : EEG Signal Detection using machine learning
dataset : kaggle dataset-- https://www.kaggle.com/datasets/shashwatwork/eeg-psychiatric-disorders-dataset
train the model with svm and knn
after that do detection on both svm and knn model to check accuracy.
from visualization, it is found that svm performs better than knn.

# EEG-Based Emotion Recognition

## Overview

This project focuses on developing a machine learning-based emotion recognition system using Electroencephalogram (EEG) brain signals. Emotion recognition from EEG data has significant applications in mental health monitoring, human-computer interaction, and neuroscience research. The project implements and compares two classification algorithms—K-Nearest Neighbors (KNN) and Support Vector Machine (SVM)—to classify human emotions into three categories: NEGATIVE, NEUTRAL, and POSITIVE based on spectral and temporal features extracted from brain activity patterns. The models are trained on a comprehensive dataset containing 2,132 EEG samples with 2,548 features per sample, incorporating signal processing techniques such as Fast Fourier Transform (FFT), statistical moments, and maximum values. The system includes comprehensive evaluation metrics, visualizations, and deployment capabilities for real-time emotion prediction from new EEG data.

## Objective

- **To develop machine learning models for emotion recognition from EEG signals**: Create classification systems that analyze brain wave patterns (EEG data) to automatically detect and classify human emotions into three categories: **NEGATIVE**, **NEUTRAL**, and **POSITIVE**. This enables non-invasive emotion analysis based on neural activity.

- **To implement and compare two classification approaches**: Build and evaluate **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** algorithms using spectral and temporal features extracted from brain activity data. This comparison helps identify the most effective approach for emotion classification.

- **To train models on comprehensive EEG dataset**: Train models on a dataset of 2,132 EEG samples with 2,548 features per sample, extracted using signal processing techniques like Fast Fourier Transform (FFT), statistical moments, and maximum values. This rich feature set captures various aspects of brain activity.

- **To evaluate model performance**: Assess classification accuracy through metrics, confusion matrices, and visualizations including confidence distributions, probability heatmaps, and error analysis. This comprehensive evaluation validates the reliability and practical applicability of the emotion recognition system.

## Work Done

**Data ingestion and validation**
- Loaded `emotions.csv` (2,132 samples × 2,548 features), checked for missing or invalid values, and handled data quality issues.
- Encoded categorical emotion labels (NEGATIVE, NEUTRAL, POSITIVE) to integers using LabelEncoder and saved the mapping for inference.

**Preprocessing**
- Fitted a StandardScaler on the training data and applied the same transformation to test data to prevent data leakage.
- Performed a stratified 80:20 train/test split (random_state=42) to preserve class proportions across splits.

**Model development**
- Implemented **K-Nearest Neighbors** (k=5, Euclidean distance) and **Support Vector Machine** (RBF kernel) using scikit-learn.
- Trained both models on scaled training data and saved predictions, class probabilities, and classification reports.
- Persisted trained models, scalers, label encoders, and feature information using joblib for deployment.

**Evaluation and error analysis**
- Calculated overall test accuracy, per-class precision/recall/F1, macro F1, and confusion matrices.
- Generated visualizations including class-wise confidence distributions, accuracy-vs-confidence curves, and probability heatmaps.
- Performed error analysis by examining confusion matrices and misclassification patterns between emotion classes.

**Detection and deployment**
- Created inference notebooks for real-time emotion prediction from new EEG data.
- Saved detection results to text files with actual labels, predicted labels, and confidence scores.

## Dataset

- **Total Samples**: 2,132
- **Features**: 2,548 per sample (mean, standard deviation, moments, max values, FFT coefficients)
- **Classes**: 3 emotion labels (NEUTRAL: 716, NEGATIVE: 708, POSITIVE: 708)

## Project Structure

```
.
├── knn/                    # KNN classifier implementation
│   ├── train_model.ipynb   # Model training
│   ├── detect.ipynb        # Emotion detection
│   └── models/             # Saved model artifacts
├── svm/                    # SVM classifier implementation
│   ├── train_model.ipynb   # Model training
│   ├── detect.ipynb        # Emotion detection
│   └── models/             # Saved model artifacts
└── archive (3)/
    └── emotions.csv        # Dataset
```

## Technologies

- Python 3.x
- scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn
- Joblib

## Usage

1. Train a model: Run `train_model.ipynb` in either `knn/` or `svm/` directory
2. Detect emotions: Use `detect.ipynb` to predict emotions from new EEG data
3. Visualize results: Run `visualization_emotion.ipynb` for analysis and plots

## Conclusion

This project successfully demonstrates the development and deployment of machine learning models for emotion recognition from EEG signals. By implementing both KNN and SVM classifiers, the system provides a comparative analysis of different algorithmic approaches for classifying emotions into negative, neutral, and positive categories. The comprehensive evaluation through accuracy metrics, confusion matrices, and visualizations validates the reliability and practical applicability of the trained models. The complete pipeline from data ingestion and preprocessing to model training, evaluation, and deployment demonstrates a systematic approach to building production-ready emotion recognition systems. The project highlights the potential of EEG-based emotion classification in real-world applications such as mental health monitoring, brain-computer interfaces, and affective computing research. Future enhancements could include exploring deep learning architectures, expanding the emotion taxonomy, and improving real-time processing capabilities.


