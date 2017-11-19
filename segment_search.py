#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# export PYOPENCL_CTX='0:1'

# Read in two images A and B, regularly segment B
# Using ORB find best correlation in A for each segment in B

# Correspondence arises from the process of elimination
# where better parts are matched on their own merit, and
# others are matched as a consequence.

# A square of exact dimension in A and B corresponds on intrinsic merit.
# A star and lambda, encased in the squares, correspond only because they are orphans.
# And because there is no intrinsic relationship between them - i.e. a default.

# The process of solving these problems is not a reasoning process that occurs subconciously
# and therefore cannot appeal to subconcious processes.

import numpy as np
import cv2
from utils.utils import segment_and_transform
from PIL import Image

max_dim = 184
abst = 46
dim = int(max_dim / abst)
segment_range = range(dim)

MIN_MATCH_COUNT = 3

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

def brute_force_matches(source_descriptors, destination_descriptors):
    """ http://opencv-python-tutroals.readthedocs.io/ """ 
    # Create BFMatcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(source, destination)

    # Sort them in the order of their distance.
    return sorted(matches, key = lambda x:x.distance)

def flan_matches(source_descriptors, destination_descriptors):
    """ http://opencv-python-tutroals.readthedocs.io/ """
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1) #2    search_params = dict(checks=50)   # or pass empty dictionary
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(source_descriptors, destination_descriptors,k=2)

    if len(matches) == 0:
        print("No matches found.")
        return [],[]

    # Need to draw only good matches, so create a mask
    #matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    #for i,(m,n) in enumerate(matches):
    #    if m.distance < 0.7*n.distance:
    #        matchesMask[i]=[1,0]
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return matches, good

img1 = cv2.imread('./images/A.png', 0) # A
img2 = cv2.imread('./images/B.png', 0) # B

# Segment and transform Images.
A, B = segment_and_transform(np.asarray(img1, dtype=np.float32), np.asarray(img2, dtype=np.float32), abst=abst)

# Initiate STAR detector - 4 pixels supports orthonomal transforms.
orb = cv2.ORB_create(WTA_K=2, nfeatures=1000, nlevels=4, scaleFactor=1.2, patchSize=4, edgeThreshold=4)

# Find keypoints in A.
A_keypoints = orb.detect(img1, None)
A_keypoints, A_descriptors = orb.compute(img1, A_keypoints)

if len(A_keypoints) == 0:
    print( 'No features detected -  no matching.')
    exit(0)

# For each segmented tranformation in B:
for index, transform in enumerate(B):
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

            print("Features found.")
            matches, good = flan_matches(A_descriptors, segment_descriptors)

            # If there are matches, determine the locality of the segment.

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([ A_keypoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ segment_keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if M is None:
                    print("No homography.")
                    continue
                matchesMask = mask.ravel().tolist()

                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                segment = cv2.polylines(segment,[np.int32(dst)],True,100,3, cv2.LINE_AA)
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                                singlePointColor = None,
                                matchesMask = matchesMask, # draw only inliers
                                flags = 2)

                img3 = cv2.drawMatches(img1,A_keypoints,segment,segment_keypoints,good,None,**draw_params)
                filename = "transform_{}_keypoints_match_{}_{}.png".format(index, i, j)
                cv2.imwrite(filename,img3)
            else:
                print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
                matchesMask = None

    # Consider a single transformation for now.