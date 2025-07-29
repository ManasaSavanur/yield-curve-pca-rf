# Yield Curve Analysis and 30-Year Yield Prediction Using PCA and Random Forest
This project demonstrates how to generate synthetic yield curve data modeled after the Nelson-Siegel framework, apply Principal Component Analysis (PCA) for dimensionality reduction, and use a Random Forest regressor to predict the 30-year yield.

# Features
📊 Synthetic Data Generation: Simulates yield curves based on level, slope, and curvature components typical in yield curve modeling.

📉 Dimensionality Reduction: Applies PCA to reduce the dimensionality of yield curve data to three principal components.

🧠 Prediction Model: Trains a Random Forest regression model to predict the 30-year yield from the PCA-transformed features.

🔍 Model Evaluation: Calculates mean squared error (MSE) on a held-out test set.

💻 Visualization: Plots actual vs. predicted 30-year yields to visually assess model performance.

# Requirements
Python 3.7+

NumPy

Pandas

Matplotlib

scikit-learn
