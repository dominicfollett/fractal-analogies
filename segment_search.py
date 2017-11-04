#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Read in two images A and B, regularly segment B
# Using ORB find best correlation in A for each segment in B

import numpy as np
import cv2
from utils.utils import segment_and_transform

img1 = np.asarray(cv2.imread('./images/1.png',0), np.float32)
img2 = np.asarray(cv2.imread('./images/4.png',0), np.float32)

# Segment Images
A, B = segment_and_transform(img1, img2)

# Initiate STAR detector
orb = cv2.ORB_create()

# For each segment in B
