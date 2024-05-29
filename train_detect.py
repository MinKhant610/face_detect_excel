import cv2 as cv
import numpy as np
import pandas as pd
from datetime import datetime

cap = cv.VideoCapture(0)
path = '/Users/minkhant/face_recon_pj/haarcascade_frontalface_alt.xml'
face_cascade = cv.CascadeClassifier(path)
face_section = np.zeros((100, 100), dtype='uint8')
face_data = []

name = input("Enter your name: ")
roll = input("Enter your roll number: ")

skip = 0 

# List to store the collected data
collected_data = []

while True:
    ret, frame = cap.read()
    
    if ret is False:
        continue 
        
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    
    for face in faces[-1:]:
        x, y, w, h = face 
        face_section = gray[y : y + h, x : x + w]
        face_section = cv.resize(face_section, (100, 100))
        cv.putText(frame, name, (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        
        # Capture the timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if skip % 10 == 0:
            # Flatten the face section and append to the collected data along with timestamp, name, and roll
            face_flat = face_section.flatten()
            collected_data.append([timestamp, name, roll] + face_flat.tolist())
        
        skip += 1 
    
    cv.imshow('Camera', frame)
    
    key = cv.waitKey(1)
    
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()

# Convert collected data to a DataFrame and save as CSV
columns = ["Timestamp", "Name", "Roll"] + [f"pixel_{i}" for i in range(100 * 100)]
df = pd.DataFrame(collected_data, columns=columns)
save_path = f'/Users/minkhant/face_recon_pj/data/{name}_{roll}_data.csv'
df.to_csv(save_path, index=False)

print(f"Data saved to {save_path}")
