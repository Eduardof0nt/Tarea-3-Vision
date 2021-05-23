import cv2
import numpy as np
import av
import time
import matplotlib.pyplot as plt

videoF = cv2.VideoCapture('video.mp4')

video = []
recY = 0.4
s = 0
if(videoF.isOpened()):
    ret, frame = videoF.read()
    s = frame.shape
    print(s)
    video.append(frame[np.floor(recY*s[0]).astype('int'):,:])
    
while(videoF.isOpened()):
    ret, frame = videoF.read()
    if not ret:
        break
    video.append(frame[np.floor(recY*s[0]).astype('int'):,:])


videoF.release()

video = np.array(video)

frameSize = (video.shape[2],video.shape[1]) #(width, height)
fps = 25
codec = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('video_roi.mp4',codec, fps, frameSize)

for frame in video:
    out.write(frame)
out.release()

show = False
while show:
    for frame in video:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            show = False
            break
