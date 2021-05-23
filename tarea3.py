import cv2
import numpy as np
import av
import time
import matplotlib.pyplot as plt

videoF = cv2.VideoCapture('video_roi.mp4')


video = []

if(videoF.isOpened()):
    ret, frame = videoF.read()
    video.append(frame)

while(videoF.isOpened()):
    ret, frame = videoF.read()
    if not ret:
        break
    
    video.append(frame)


videoF.release()

video = np.array(video)

# while True:
#     for frame in video:
#         cv2.imshow('frame',frame)
#         time.sleep(33)

videoF = []

# for i, frame0 in enumerate(video):
#     Z = frame0.reshape((-1,3))
#     Z = np.float32(Z)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #Max iter = 10, epsilon=1.0
#     K = 20
#     ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     res2 = res.reshape((frame0.shape))
#     videoF.append(res2);
#     cv2.imshow('Frame',res2)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
    

show = True
while show:
    for frame0 in video:
        cv2.imshow('Frame',frame0)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            show = False
            break
        elif key == ord(' '):
                # K-Means
                Z = frame0.reshape((-1,3))
                Z = np.float32(Z)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #Max iter = 10, epsilon=1.0
                K = 15
                ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                res = center[label.flatten()]
                res2 = res.reshape((frame0.shape))
                
                # Segmentación
                img = cv2.cvtColor(res2, cv2.COLOR_BGR2HSV)

                mask1 = cv2.inRange(img, (90,100,50), (130,255,255))
                # mask2 = cv2.inRange(img, lower2, upper2)
                
                full_mask = mask1 #+ mask2
                
                res3 = cv2.bitwise_and(res2, res2, mask=full_mask)
                
                full_mask = np.uint8(full_mask)
                
                plt.imshow(full_mask, cmap="gray")
                plt.show(block=False)
                cv2.imshow('Frame',res3)
                cv2.waitKey(0)
                continue

# plt.imshow(videoF[100], cmap = plt.cm.gray)
# plt.show()

# cv2.imshow('frame',videoD[0])
cv2.destroyAllWindows()
