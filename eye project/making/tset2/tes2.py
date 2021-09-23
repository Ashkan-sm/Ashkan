import cv2
import numpy as np
im1=cv2.imread('1.png')
im2=cv2.imread('2.png')

pts1 = np.float32([[0,0],[0,300],[400,0],[400,300]])
pts2 = np.float32([[0,0],[0,150],[400,0],[400,150]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(im1,M,(800,600))
cv2.imshow('1',dst)
cv2.waitKey()