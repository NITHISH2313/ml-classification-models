#model_evaluation


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, 
    confusion_matrix, classification_report)

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

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# ===== TRAIN =====
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===== EVALUATION =====
print("=" * 40)
print("      MODEL EVALUATION REPORT")
print("=" * 40)

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")

# 2. confusion matrix

cm = confusion_matrix(y_test, y_pred)
print(f"\n confusion matrix :{cm}")
print(f"\n true negatives (predicted fail , actually fail): {cm[0][0]}")
print(f"\n true positives (predicted pass , actually pass): {cm[1][1]}")
print(f"\n false negatives (predicted pass , actually fail): {cm[1][0]}")
print(f"\n false negatives (predicted fail , actually pass): {cm[0][1]}")

print(f"\n classification report")
print(classification_report(y_test,y_pred,target_names=["Fail" , "Pass"]))

