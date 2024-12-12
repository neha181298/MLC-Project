# Online News Popularity Prediction

## Overview
This project aims to predict the popularity of online news articles, measured by the number of shares an article receives. Various supervised machine learning models were implemented to analyze relationships between article features—such as content, metadata, and sentiment—and their impact on popularity. This repository includes the complete workflow, from preprocessing and feature engineering to model training and evaluation.

## Repository Structure
- **`MLC_Final_Project_Pipeline.ipynb`**: Main pipeline for data preprocessing, feature engineering, model training, and evaluation.
- **`MLC_Final_Project_Experimentation.ipynb`**: Details experimental setups, hyperparameter tuning, and model comparisons.
- **`modules.py`**: Custom utility functions for data processing and machine learning tasks.
- **`requirements.txt`**: Lists all dependencies required to reproduce the project.
- **Dataset**: Online News Popularity Dataset from UCI Machine Learning Repository.

## Key Features
1. **Data Preprocessing**:
   - Removal of outliers and invalid data.
   - Feature selection based on correlation and redundancy analysis.
   - Handling skewed distributions using capping methods.

2. **Modeling**:
   - Implementation of multiple regression models:
     - Linear Regression
     - Ridge Regression
     - Random Forest Regressor
     - XGBoost Regressor
     - Feed Forward Neural Network
   - Evaluation metrics include:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - R²
     - Mean Absolute Percentage Error (MAPE)

3. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA) was applied, retaining 95% of explained variance.

4. **Results**:
   - XGBoost Regressor consistently achieved the best performance metrics.
   - PCA did not significantly improve results, indicating the importance of feature selection.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
