import cv2
import numpy as np
from keras.models import load_model

# laptop camera
rgb = cv2.VideoCapture(0)


# pre - trinaed xml file for detecting faces
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

model = load_model('face_reco.h5')
while True:
    ret, fr = rgb.read()
    flip_fr = cv2.flip(fr,1)
    if ret is True:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    else:
        continue
    faces = facec.detectMultiScale(gray, 1.2,5)
    
    for (x,y,w,h) in faces:
        fc = fr[y:y+h, x:x+w, :]
        im = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (48, 48))
        im = im[np.newaxis, np.newaxis, :, :]
        res = model.predict_classes(im,verbose=0)
        emo = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        flip_fr = cv2.flip(fr,1)
        cv2.putText(flip_fr, emo[res[0]], (30, 30), font, 1, (255, 0,255), 5)
    
    cv2.imshow('rgb', flip_fr)

    # press esc to close the window
    k = cv2.waitKey(1) & 0xEFFFFF
    if k==27:   
        break
    elif k==-1:
        continue
    else:
        # print k
        continue

rgb.release()
cv2.destroyAllWindows()
