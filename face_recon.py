import cv2 as cv
import numpy as np
import pandas as pd
from datetime import datetime  # Added datetime import
import os

cap = cv.VideoCapture(0)
path = '/Users/minkhant/face_recon_pj/haarcascade_frontalface_alt.xml'
face_cascade = cv.CascadeClassifier(path)
data_path = '/Users/minkhant/face_recon_pj/data/'

face_data = []
labels = []
name_roll = {}
class_id = 0

def distance(x, X):
    return np.sqrt(np.sum((x - X) ** 2))

def knn(X, Y, x, K=5):
    m = X.shape[0]
    x = x.flatten()
    val = []
    for i in range(m):
        xi = X[i].flatten()
        dist = distance(x, xi)
        val.append((dist, Y[i][0]))  # Ensure labels are scalars
    
    val = sorted(val, key=lambda x: x[0])[:K]
    
    # Convert val to numpy array
    val = np.array(val)
    
    # Get the labels from val
    closest_labels = val[:, 1]
    
    # Find the most common label in the closest K labels
    unique_labels, counts = np.unique(closest_labels, return_counts=True)
    predicted_label = unique_labels[np.argmax(counts)]

    return predicted_label

# Load training data from CSV files
for file in os.listdir(data_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_path, file))
        face_data.append(df.iloc[:, 3:].values)  # Load face pixel data (excluding Timestamp, Name, Roll)
        name_roll[class_id] = (df.iloc[0, 1], df.iloc[0, 2])  # Load Name and Roll
        target = class_id * np.ones((df.shape[0],), dtype=int)
        class_id += 1
        labels.append(target)

# Concatenate all the training data
face_data_set = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape(-1, 1)

# Initialize the list to store detected names, rolls, and timestamps
detected_names_rolls = []

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    for face in faces[-1:]:
        x, y, w, h = face
        face_section = gray[y:y + h, x:x + w]
        face_section = cv.resize(face_section, (100, 100))

        predict = knn(face_data_set, face_labels, face_section)
        predict_name, predict_roll = name_roll[int(predict)]
        
        # Add the detected name, roll, and timestamp to the list
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        detected_names_rolls.append((timestamp, predict_roll, predict_name))

        cv.putText(frame, f'{predict_name} ({predict_roll})', (x, y - 30), cv.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0), 2, cv.LINE_AA)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))

    cv.imshow('Camera', frame)

    key = cv.waitKey(1)

    if key == 27:
        break

cap.release()
cv.destroyAllWindows()

# Save the detected names, rolls, and timestamps to an Excel file
df = pd.DataFrame(detected_names_rolls, columns=["TImestamp", "Roll", "Name"])
df.to_excel('detected_names.xlsx', index=False)
print("Detected names, rolls, and timestamps saved to 'detected_names.xlsx'")
