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

def get_segment(i, j, abst, image):
    """:returns: a segment (i,j) for an abstraction level in the given image.
    
    warning:: Image shape must be square, and dimensions evenly divisible by abst.

    :param i: matrix row index
    :param j: matrix column index
    :param abst: the level of abstraction under consideration
    :param image: the image to segment
    :type i: int 
    :type j: int
    :type abst: int
    :type image: 2D numpy array
    """
    istart, iend = i * abst, (i+1) * abst
    jstart, jend = j * abst, (j+1) * abst
    return np.asarray(transform[istart:iend, jstart:jend], dtype=np.uint8)

def find_brute_force_matches(source, destination):
    # Create BFMatcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(source, destination)

    # Sort them in the order of their distance.
    return sorted(matches, key = lambda x:x.distance)

img1 = cv2.imread('./images/1.png', 0) # A
img2 = cv2.imread('./images/4.png', 0) # B

# Segment and transform Images.
A, B = segment_and_transform(np.asarray(img1, dtype=np.float32), np.asarray(img2, dtype=np.float32), abst=abst)

# Initiate STAR detector - 4 pixels supports orthonomal transforms.
orb = cv2.ORB_create(WTA_K=2, nfeatures=1000, nlevels=8, scaleFactor=1.2, patchSize=4, edgeThreshold=4)

# Find keypoints in A.
A_keypoints = orb.detect(img1, None)
A_keypoints, A_descriptors = orb.compute(img1, A_keypoints)

# For each segmented tranformation in B:
for transform in B:
    for i in segment_range:
        for j in segment_range:
            # For each segment:
            segment = get_segment(i, j, abst, transform)

            # find the keypoints in segment with ORB
            segment_keypoints = orb.detect(segment, None)

            # compute the descriptors with ORB
            segment_keypoints, segment_descriptors = orb.compute(segment, segment_keypoints)

            if len(segment_keypoints) == 0:
                print( 'No features detected -  no matching.')
                continue

            # If there are matches, determine the locality of the segment.

            # Draw first 10 matches.
            #img3 = cv2.drawMatches(img1,A_keypoints,segment,segment_keypoints,matches[:100], flags=2, outImg=None)

            #filename = "{}_keypoints_match_{}.png".format(i, j)
            #cv2.imwrite(filename,img3)
    # Consider a single transformation for now.
    exit(0)
