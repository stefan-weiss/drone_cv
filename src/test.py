import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newimg = gray.copy()

    sift = cv2.SIFT()
    kp = sift.detect(gray,None)

    img =cv2.drawKeypoints(gray,kp)

    #laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    #sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    #sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    edges = cv2.Canny(gray,100,200)

    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(newimg,(x,y),3,255,-1)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    #cv2.imshow('laplacian', laplacian)
    #cv2.imshow('sobelx', sobelx)
    #cv2.imshow('sobely', sobely)
    cv2.imshow('edges', edges)
    cv2.imshow('corners', newimg)
    cv2.imshow('test', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
