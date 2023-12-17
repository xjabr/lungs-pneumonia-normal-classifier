import pickle

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from glob import glob

from dataset import load_dataset

X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SGDClassifier(verbose=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(f"ERROR -> Actual: {y_test[i]} Predicted: {y_pred[i]}")

print("Total number of test examples:", len(y_test))
print("Total number of correct predictions:", np.sum(y_test == y_pred))
print("Total number of errors:", len(y_test) - np.sum(y_test == y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))