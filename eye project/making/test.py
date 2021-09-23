import cv2
img=cv2.imread('1.jpg')
cv2.imshow('1',img[:200,0:100])
cv2.waitKey()