# Read in two images A and B, regularly segment B
# Using ORB find best correlation in A for each segment in B

import numpy as np
import cv2

img1 = cv2.imread('./images/1.png',0)
img2 = cv2.imread('./images/4.png',0)

# Initiate STAR detector
orb = cv2.ORB_create()

# Segment B


# For each segment in B
