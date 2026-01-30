import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()
X = iris.data   # features (sepal length, petal length, etc.)
y = iris.target # labels (flower species)

print("Shape of X:", X.shape)
print("First 5 rows of X:\n", X[:5])
print("First 5 labels:", y[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred[:10])
print("Actual     :", y_test[:10])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
