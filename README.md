# Melbourne House Price Prediction

## Overview

This project predicts suburban house prices in Melbourne using machine learning.
It uses five core features:

- Suburb
- Rooms
- Bathroom
- Car
- Distance from CBD

Multiple regression models were trained and evaluated, and Random Forest was
selected for its superior performance.

This project was originally developed in 2020 as part of Machine Learning course at VIT Vellore.

Recently refactored and uploaded for documentation and portfolio.

## Dataset

Source: Kaggle – Melbourne Housing Market Dataset

File: Melbourne_housing_FULL.csv

## Models Used

- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Evaluation Metrics

- R²
- RMSE
- MAE

## Results

After feature expansion and hyperparameter tuning, the models achieved:

| Model            | R² Score |
|------------------|----------|
| Linear Regression| 0.68     |
| Ridge Regression | 0.69     |
| Decision Tree    | 0.63     |
| Random Forest    | 0.81     |
| Gradient Boosting| 0.81     |

## Tech Stack

Python, pandas, scikit-learn, Jupyter, matplotlib, seaborn

## How to Run

```bash
pip install -r requirements.txt
cd src
python model_training.py