import datetime as dt

print("hello!")
pyscript.write("date",dt.datetime.today().strftime("%A %B %d ,%Y"))
pyscript.write("name", "aryan dixit")

import cv2
from cvzone.HandTrackingModule import HandDetector
# from directkeys import Pressingkey
# from directkeys import space_pressed
import time

detector=HandDetector(detectionCon=0.8, maxHands=1)
video=cv2.VideoCapture(0)
ret,frame=video.read()
hands,img=detector.findHands(frame)
cv2.imshow("Frame",frame)

video.release()
cv2.destroyAllWindows()
# space_key_pressed=space_pressed

time.sleep(2.0)
