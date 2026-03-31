#cross_validation

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split,
    cross_val_score, GridSearchCV)
from sklearn.metrics import accuracy_score

# ===== DATA =====
X = np.array([
    [2, 60, 45], [3, 70, 55], [4, 75, 60],
    [5, 80, 65], [6, 85, 70], [7, 90, 78],
    [8, 92, 82], [9, 95, 88], [10, 98, 95],
    [1, 50, 35], [3, 65, 52], [5, 78, 68],
    [7, 88, 76], [8, 91, 84], [6, 83, 71],
    [4, 72, 58], [9, 94, 90], [2, 55, 42],
    [5, 75, 63], [7, 85, 74]
])

y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1,
              0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1])

# ===== CROSS VALIDATION =====
model = DecisionTreeClassifier()

# Test model 5 times on different splits!
scores = cross_val_score(model, X, y, cv=5)

print("===== CROSS VALIDATION =====")
print(f"Scores each fold: {scores.round(2)}")
print(f"Mean accuracy:    {scores.mean()*100:.2f}%")
print(f"Std deviation:    {scores.std()*100:.2f}%")

# ===== HYPERPARAMETER TUNING =====
# Find the BEST settings for your model!
param_grid = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X, y)

print("\n===== HYPERPARAMETER TUNING =====")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy:   {grid_search.best_score_*100:.2f}%")

# ===== BEST MODEL =====
best_model = grid_search.best_estimator_
print(f"\nBest model: {best_model}")

