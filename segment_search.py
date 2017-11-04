#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# export PYOPENCL_CTX='0:1'

# Read in two images A and B, regularly segment B
# Using ORB find best correlation in A for each segment in B

import numpy as np
import cv2
from utils.utils import segment_and_transform
from PIL import Image

max_dim = 184
abst = 92
dim = int(max_dim / abst)
segment_range = range(dim)


img1 = cv2.imread('./images/1.png',0) # A
img2 = cv2.imread('./images/4.png',0) # B

# Segment and transform Images
A, B = segment_and_transform(np.asarray(img1, np.float32), np.asarray(img2, np.float32), abst=abst)

# Initiate STAR detector
orb = cv2.ORB_create()

# Find keypoints in A
A_keypoints = orb.detect(img1, None)
A_keypoints, A_descriptors = orb.compute(img1, A_keypoints)

# For each segmented tranformation in B
for transform in B:
    for i in segment_range:
        for j in segment_range:
            istart, iend = i * abst, (i+1) * abst
            jstart, jend = j * abst, (j+1) * abst
            segment = transform[istart:iend, jstart:jend]

            # find the keypoints in segment with ORB
            segment_keypoints = orb.detect(segment, None)

            # compute the descriptors with ORB
            segment_keypoints, segment_descriptors = orb.compute(segment, segment_keypoints)

            if len(segment_keypoints) == 0:
                print( 'No features detected.')
                exit(0)

            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors.
            matches = bf.match(A_descriptors,segment_descriptors)

            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)

            # Draw first 10 matches.
            img3 = cv2.drawMatches(img1,A_keypoints,segment,segment_keypoints,matches[:10], flags=2, outImg=None)

            cv2.imwrite('orb_keypoints_match.jpg',img3)

            exit(0)

