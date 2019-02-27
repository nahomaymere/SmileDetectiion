import numpy as np
import cv2
from keras.models import load_model

cap = cv2.VideoCapture(0)
model = load_model('model.h5')
counter = 0

while(True):
    # Capture frame-by-frame
    ret, initial_frame = cap.read()
    # Our operations on the frame come here
    face_cascade = cv2.CascadeClassifier('C:\\Users\\aymer\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    pred = 0
    for(x,y,w,h) in faces:
        cv2.rectangle(initial_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = initial_frame[y:y + h, x:x + w]
        roi_color = cv2.resize(roi_color,dsize=(64,64), interpolation = cv2.INTER_CUBIC)
        rgbFace = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        rgbFace = np.reshape(rgbFace, (1, 64, 64, 3))
        pred = model.predict_classes(np.array(rgbFace))
        print(pred)
    if(pred == 1):
        cv2.putText(initial_frame,'smile',(230, 50), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(initial_frame,'no smile',(230, 50), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('nahom',initial_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()