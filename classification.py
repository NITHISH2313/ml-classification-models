#classification and randomforest 

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#=======data======
# hours studied ,attendance ,previous score

X = np.array([
    [2, 60, 45], [3, 70, 55], [4, 75, 60],
    [5, 80, 65], [6, 85, 70], [7, 90, 78],
    [8, 92, 82], [9, 95, 88], [10, 98, 95],
    [1, 50, 35], [3, 65, 52], [5, 78, 68],
    [7, 88, 76], [8, 91, 84], [6, 83, 71],
    [4, 72, 58], [9, 94, 90], [2, 55, 42],
    [5, 75, 63], [7, 85, 74]
])

# Pass = 1, Fail = 0
Y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1,
              0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1])

X_train ,X_test,Y_train,Y_test = train_test_split(X ,Y ,test_size=0.2 , random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
print("model trainied")

#====== test=======
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)
print(f"accuracy :{accuracy*100:.2f}")


new_student = np.array([[7,85,72]])
prediction = model.predict(new_student)
result = "pass" if prediction[0] == 1 else "fail"
print(f"prediction :{result}")


model_rf = RandomForestClassifier(n_estimators = 100)
model_rf.fit(X_train,Y_train)

Y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(Y_test , Y_pred_rf)

features = ["hours " ,"attendance" , "prev_score"]
importance = model_rf.feature_importances_

print(f"\n feature importance")
for f , i in zip(features , importance):
    print(f"{f}: {i*100:.1f}%")
