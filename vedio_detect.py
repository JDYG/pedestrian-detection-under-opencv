#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 22:26:03 2018

@author: gsy
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import time


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
#    print("Time for NMS_fast:           {} seconds".format(NMSEnd - NMSStart)) 
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
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
defaultHOG = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                               derivAperture,winSigma,histogramNormType,
                               L2HysThreshold,gammaCorrection,nlevels)
defaultHOG.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())


cv2.namedWindow("Vedio", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("F:/graduate_0/mypackges/myPedestrianDection/myHOGwork/mypicture/V013.mkv")
n = 0
while(True):
    start = time.time()
    ret, frame = cap.read()
    n += 1
    if n % 3 != 0:
        continue
    realTestImg = frame
    (rects, weights) = defaultHOG.detectMultiScale(realTestImg, winStride = winStride, padding=(8, 8), scale=1.01, useMeanshiftGrouping = False)
    nms_rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_maximum_supression_fast(nms_rects, 0.3)
    for (x1, y1, x2, y2) in pick:
        cv2.rectangle(realTestImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    end = time.time()
    print("THe time per frame:{} seconds".format(end - start))
    cv2.imshow("Vedio", realTestImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()

cv2.waitKey(0) 
cv2.destroyAllWindows()