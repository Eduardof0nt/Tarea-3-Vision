import cv2
import numpy as np

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

videoF = []

show = True
while show:
    for frame0 in video:
        cv2.imshow('Frame',frame0)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            show = False
            break
        elif key == ord(' '):
                frame = frame0.copy()
                
                # K-Means
                Z = frame0.reshape((-1,3))
                Z = np.float32(Z)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #Max iter = 10, epsilon=1.0
                K = 40
                ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                res = center[label.flatten()]
                res2 = res.reshape((frame0.shape))
                
                # Segmentación
                img = cv2.cvtColor(res2, cv2.COLOR_BGR2HSV)

                low = (100,110,30)
                high = (170,255,180)

                mask1 = cv2.inRange(img, low, high)
                
                full_mask = mask1 #+ mask2
                
                full_mask = np.uint8(full_mask)
                
                blur = cv2.medianBlur(full_mask,13)
                
                contours, hierarchy = cv2.findContours(blur.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                
                valid_cntrs = []
                for cntr in contours:
                    x,y,w,h = cv2.boundingRect(cntr)
                    if cv2.contourArea(cntr) >= 200 and cv2.contourArea(cntr) <= 8000 and h/w < 1.8 and h/w > 0.5:
                        valid_cntrs.append(cntr)
                        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
                cv2.imshow('Frame',frame)
                # cv2.imshow('Mask',full_mask)
                # cv2.imshow('Mask-Blur',blur)
                # cv2.imshow('K-Means',res2)
                
                # lo_square = np.full((100, 100, 3), low, dtype=np.uint8)
                # do_square = np.full((100, 100, 3), high, dtype=np.uint8)
                # cv2.imshow('Low',cv2.cvtColor(lo_square, cv2.COLOR_HSV2RGB))
                # cv2.imshow('High',cv2.cvtColor(do_square, cv2.COLOR_HSV2RGB))
                
                cv2.waitKey(0)
                # cv2.destroyWindow('Mask')
                # cv2.destroyWindow('Mask-Blur')
                # cv2.destroyWindow('K-Means')
                # cv2.destroyWindow('Low')
                # cv2.destroyWindow('High')
                continue

cv2.destroyAllWindows()
