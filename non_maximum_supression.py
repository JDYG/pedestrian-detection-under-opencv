#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:48:37 2018

@author: gsy
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

def non_maximum_supression_slow(boxes, overlapThresh):
    NMSStart = time.time()
    if len(boxes) == 0:
        return 0
    # initialize the list of picked indexes
    pick = []
 
	# grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(x2)
    # keep looping while some indexes still remain in the indexes
	# list
    while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
			# grab the current index
            j = idxs[pos]
 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
 
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the
			# current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
#                print("boxes:{}\n".format(boxes[i]))
#                print("supress:{}\n".format(boxes[j]))
		# delete all indexes from the index list that are in the
		# suppression list
        idxs = np.delete(idxs, suppress)
    NMSEnd = time.time()    
	# return only the bounding boxes that were picked
    print("Time for NMS_slow:           {} seconds".format(NMSEnd - NMSStart))
    return boxes[pick]    


def non_maximum_supression_fast(boxes, overlapThresh):
    NMSStart = time.time()
	# if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
    pick = []
 
	# grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(x2)
 
	# keep looping while some indexes still remain in the indexes
	# list
    while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate( ([last], \
                      np.where(overlap > overlapThresh)[0]) ))
    NMSEnd = time.time()
    print("Time for NMS_fast:           {} seconds".format(NMSEnd - NMSStart)) 
	# return only the bounding boxes that were picked using the
	# integer data type
    return boxes[pick].astype("int")

winStride = (8, 8) #This paramenter controls the winStride in defaultHOG.detectMultiScale
## Initial the HOGDescriptor
winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -0.1
histogramNormType = 1
L2HysThreshold = 0.8
gammaCorrection = True
nlevels = 1000
defaultHOG = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                               derivAperture,winSigma,histogramNormType,
                               L2HysThreshold,gammaCorrection,nlevels)
defaultHOG.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

inputImg = cv2.imread("F:/graduate_0/mypackges/myPedestrianDection/myHOGwork/mypicture/biandianzhan1.jpg")
nmsImg = inputImg.copy()

detectImgStart = time.time()
realTestImg = inputImg
(rects, weights) = defaultHOG.detectMultiScale(realTestImg, winStride = winStride, padding=(8, 8), scale=1.03, useMeanshiftGrouping = False)
detectImgEnd = time.time()
print("Time for detection(default): {} seconds".format(detectImgEnd - detectImgStart))
for (x, y, w, h) in rects:
		cv2.rectangle(realTestImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("Befor NMS", realTestImg)

nms_rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_maximum_supression_fast(nms_rects, 0.5)
for (x1, y1, x2, y2) in pick:
		cv2.rectangle(nmsImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("After NMS", nmsImg)        

cv2.waitKey(0)
cv2.destroyAllWindows()