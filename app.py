import pickle
import numpy as np

from glob import glob

from dataset import image_preprocessing

model = pickle.load(open("model.sav", 'rb'))

val_images_normal = glob('datasets/val/NORMAL/*.jpeg')
val_images_pneumonia = glob('datasets/val/PNEUMONIA/*.jpeg')
val_label_normal = [0] * len(val_images_normal)
val_label_pneumonia = [1] * len(val_images_pneumonia)

x, y = list(val_images_normal + val_images_pneumonia), list(val_label_normal + val_label_pneumonia)

X_val = np.stack([image_preprocessing(image_path) for image_path in x], axis=0)
y_val = np.array(y)

y_pred = model.predict(X_val)

for i in range(len(y_val)):
    if y_val[i] != y_pred[i]:
        print(f"ERROR -> Actual: {y_val[i]} Predicted: {y_pred[i]}")