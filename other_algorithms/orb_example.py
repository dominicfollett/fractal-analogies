import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../images/2.png', 0)
img2 = cv2.imread('../images/1.png', 0)

# Initiate STAR detector
orb = cv2.ORB_create(WTA_K=2, nfeatures=1000, nlevels=8, scaleFactor=1.2, patchSize=31, edgeThreshold=31)

# find the keypoints with ORB
kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)


if len(kp1) == 0:
    print( 'No features detected.')
    exit(0)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2, outImg=None)

cv2.imwrite('orb_keypoints_match.jpg',img3)

# draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
# cv2.imwrite('orb_keypoints.jpg',img2)
# plt.imshow(img2),plt.show()