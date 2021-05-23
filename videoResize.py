import cv2
import numpy as np
import av
import time
import matplotlib.pyplot as plt

videoF = cv2.VideoCapture('video.mp4')

video = []
recY = 0.4

if(videoF.isOpened()):
    ret, frame = videoF.read()
    s = frame.shape
    print(s)
    video.append(frame[np.floor(recY*s[0]).astype('int'):,:])
    
while(videoF.isOpened()):
    ret, frame = videoF.read()
    if not ret:
        break
    s = frame.shape
    video.append(frame[np.floor(recY*s[0]).astype('int'):,:])


videoF.release()

video = np.array(video)

show = True
while show:
    for frame in video:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            show = False
            break
