import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../models/gesture_model.h5")

IMG_SIZE = 64

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    
    prediction = model.predict(img)
    label = np.argmax(prediction)
    
    cv2.putText(frame, f"Gesture: {label}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()