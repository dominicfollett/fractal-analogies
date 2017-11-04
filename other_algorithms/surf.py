import cv2
import numpy as np

img = cv2.imread('./images/1.png',0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.SURF(34500)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

cv2.imwrite('sift_keypoints.jpg',img2)
