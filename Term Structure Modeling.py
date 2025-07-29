import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic yield curve data
np.random.seed(42)
n_samples = 500
maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])
n_maturities = len(maturities)

# Nelson-Siegel-like components: level, slope, curvature
level = np.random.normal(0.03, 0.005, n_samples)
slope = np.random.normal(-0.01, 0.003, n_samples)
curvature = np.random.normal(0.005, 0.002, n_samples)

# Synthetic yields
yields = []
for i in range(n_samples):
    term_structure = (level[i]
                      + slope[i] * np.exp(-maturities / 3)
                      + curvature[i] * ((maturities / 3) * np.exp(-maturities / 3)))
    noise = np.random.normal(0, 0.0005, n_maturities)
    yields.append(term_structure + noise)

yields = np.array(yields)

# PCA to reduce dimensionality
pca = PCA(n_components=3)
X = pca.fit_transform(yields)
y = yields[:, -1]  # Predict 30-year yield

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Plot predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual 30Y Yield")
plt.ylabel("Predicted 30Y Yield")
plt.title("Random Forest: Predicting 30-Year Yield from PCA Components")
plt.grid(True)
plt.tight_layout()
plt.show()
