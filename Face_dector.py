import cv2 as cv
import numpy as np
#######################
face_cas = cv.CascadeClassifier("")
#######################

cap = cv.VideoCapture(1)


while True:
    success, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    feces = face_cas.detetMultiScale(frame_gray, 1.3 , 5)
    
    for (x ,y ,w ,h) in feces :
        cv.rectangle(frame, (x,y), (x+w ,y+h), (255,0,255), 2)
        
        
    
    
    
    
    
    
    
    cv.imshow("Result", frame)
    
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    
    

cv.release()    