# Heart Disease Prediction - Machine Learning Application

## Problem Statement

Cardiovascular disease remains the leading cause of death globally. Early detection and risk assessment are critical for prevention and timely intervention.

This project attempts to **predict the presence of heart disease using clinical measurements** through machine learning techniques. By analyzing patient demographics, vital signs, and cardiac test results, we aim to build an accurate predictive model that can **provide interpretable insights** into which clinical factors are most strongly associated with heart disease

## Dataset Overview

The dataset contains **918 patient records** with 11 clinical features and 1 target variable:

**Clinical Features:**
- Age
- Sex
- Chest Pain Type (Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic)
- Resting Blood Pressure
- Cholesterol levels
- Fasting Blood Sugar (> 120 mg/dl)
- Resting ECG results
- Maximum Heart Rate achieved
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- ST Slope

**Target Variable:**
- Heart Disease (0 = Normal, 1 = Heart Disease)

## Project Goals

- Build and evaluate multiple ML models (Logistic Regression, Random Forest, XGBoost, Neural Networks, etc.)
- Achieve high accuracy while maintaining strong recall (minimizing false negatives)
- Provide feature importance analysis to understand key risk factors
- Create a user-friendly interface for real-time risk assessment
- Deploy the model as a web application for clinical use
