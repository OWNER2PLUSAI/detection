import cv2 as cv
import numpy as np
#######################
face_cas = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
smaile_cas = cv.CascadeClassifier("haarcascade_smile.xml")
eyes_cas = cv.CascadeClassifier("haarcascade_eye.xml")
#######################

cap = cv.VideoCapture(1)


while True:
    success, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # feces = face_cas.detectMultiScale(frame_gray, 1.3 , 5)
    
    
    
    smile = smaile_cas.detectMultiScale(frame_gray)
    for (sx ,sy ,sw ,sh) in smile:
        cv.rectangle(frame, (sx,sy), (sx+sw ,sy+sh), (0,0,255), 1)
        
    
    # for (x ,y ,w ,h) in feces :
    #     cv.rectangle(frame, (x,y), (x+w ,y+h), (255,0,255), 2)
        
    #     frame_gra_roi = frame_gray[y:y+h, x:x+w]
    #     frame_roi = frame_gray[y:y+h, x:x+w]
        
        
        # eyes = eyes_cas.detectMultiScale(frame_roi)
        # for (ex ,ey ,ew ,eh) in smile:
        #     cv.rectangle(frame, (ex,ey), (ex+ew ,ey+eh), (255,0,255), 1)
    
    cv.imshow("Result", frame)
    
    if cv.waitKey(5) & 0xFF == ord("q"):
        break
    
    
cv.destroyAllWindows()
