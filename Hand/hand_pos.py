import cv2 as cv
from cvzone.HandTrackingModule import HandDetector 
import time


cap = cv.VideoCapture(1)
detector = HandDetector(maxHands=2 ,detectionCon= 0.6)
##### FPS
prev_frame_time = 0
new_frame_time = 0
###############

while True:
    rec ,frame = cap.read()
    hand ,frame = detector.findHands(frame)
    if hand:
        hand1 = hand[0]
        lmlist_right = hand1["lmList"] 
        lengh_right ,info_r ,frame = detector.findDistance(lmlist_right[4][:-1], lmlist_right[8][:-1] ,frame)
        text_dis_left = cv.putText(frame ,f"Right : {lengh_right :.2f}" ,(50,100) ,cv.FONT_HERSHEY_SIMPLEX ,1 ,(255,0,0) ,2)
          
               
    ######### show fps   
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time   
    text_fps = cv.putText(frame ,f"Fps : {fps :.2f}" ,(50,50) ,cv.FONT_HERSHEY_SIMPLEX , 1 ,(0,255,0) ,2)
    
    cv.imshow("Result", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cv.destroyAllWindows()