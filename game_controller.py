import os
import cv2
import time
import pyautogui
cap = cv2.VideoCapture(0)


while True:
    ret, image = cap.read()
    cv2.imshow('cam picture', image) 
    #time.sleep(0.5)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    pyautogui.press("k")


