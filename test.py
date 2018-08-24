# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from matplotlib import pyplot as plt
import glob
import cv2
import time
print('someoneoneoneone else')   

#
#data_dir = '/home/gsy/Documents/myPedeatrainDetestration/Dataset'
#
##90_160 pos
#posNum_90_160 = len( glob.glob( "{}/pos/*.png".format(data_dir) ) )
##64_128 neg
#negNum_64_128 = len( glob.glob( "{}/neg/*.png".format(data_dir) ) )
### initial hog histMat and lable size
#tempName = "/home/gsy/Documents/myPedeatrainDetestration/Dataset/pos/crop001001a.png"
#tempImg = cv2.imread(tempName)
#tempImg = tempImg[16:16+128, 16:16+64]
#winSize = (tempImg.shape[1], tempImg.shape[0])
#blockSize = (16,16)
#blockStride = (8,8)
#cellSize = (8,8)
#nbins = 9
#derivAperture = 1
#winSigma = 4.
#histogramNormType = 0
#L2HysThreshold = 2.0000000000000001e-01
#gammaCorrection = 0
#nlevels = 64
#hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
#                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#tempHist = hog.compute(tempImg)
#sampleFeatureMat = np.zeros((posNum_90_160 + negNum_64_128, len(tempHist)))
#sampleLabelMat = np.ones((posNum_90_160 + negNum_64_128, 1))
#
#startTime = time.time()
### calculate pos pictures hog
#for imgName, i in zip( glob.glob( "{}/pos/*.png".format(data_dir) ), np.arange(posNum_90_160) ):
#    tempImg = cv2.imread(imgName)
#    tempImg = tempImg[16:16+128, 16:16+64]
#    tempHist = hog.compute(tempImg)
#    sampleFeatureMat[i, :] = tempHist.reshape(1, -1)
#    sampleLabelMat[i, :] = 1
### calculate neg pictures hog
#for imgName, i in zip( glob.glob( "{}/neg/*.png".format(data_dir) ), np.arange(negNum_64_128) ):
#    tempImg = cv2.imread(imgName)
#    tempHist = hog.compute(tempImg)
#    sampleFeatureMat[i+posNum_90_160, :] = tempHist.reshape(1, -1)
#    sampleLabelMat[i+posNum_90_160, :] = -1
#endReadTime = time.time()
#print("Read Data time:{} seconds".format(endReadTime - startTime))    
#
#print("Start train SVM...")
#svm = cv2.ml.SVM_create()
#svm.setGamma(6.1)
#svm.setC(2.2)
#svm.setKernel(cv2.ml.SVM_LINEAR)
#svm.setType(cv2.ml.SVM_C_SVC)
#svm.train(np.float32(sampleFeatureMat), cv2.ml.ROW_SAMPLE, np.int32(sampleLabelMat))
#endTrainTime = time.time()
#print("Train time:{} seconds".format(endTrainTime - endReadTime))
##cc = svm.predict( np.float32(sampleFeatureMat[10:20, :]) )[1].ravel()
#alphaMat = svm.getDecisionFunction(0)[1]
#supportVectorMat = svm.getSupportVectors()
#
#resultMat = -1*alphaMat.dot(supportVectorMat)
#myDetector = resultMat.copy()
#rho = svm.getDecisionFunction(0)[0]
#myDetector = np.append(myDetector, rho)
#np.savetxt("myDetector2.txt", myDetector)

winStride = (8, 8)

   
#myDetector = np.loadtxt("myDetector2.txt")
#myDetector = myDetector[:, np.newaxis]
#winSize = (64, 128)
#blockSize = (16, 16)
#blockStride = (8, 8)
#cellSize = (8, 8)
#nbins = 9
#derivAperture = 1
#winSigma = -0.1
#histogramNormType = 0
#L2HysThreshold = 0.2
#gammaCorrection = 1
#nlevels = 64
#myHOG = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
#                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#myHOG.setSVMDetector(myDetector)
#detectImgStart = time.time()
#testImg = cv2.imread("/home/gsy/Documents/myPedeatrainDetestration/Dataset/INRIAPerson/Test/pos/person_064.png")
##print("Processing MultiScale detection")
#print(winStride)
#(rects, weights) = myHOG.detectMultiScale(testImg, winStride = winStride, padding=(8, 8), scale=1.02, useMeanshiftGrouping = False)
#detectImgEnd = time.time()
#print("Time of detection(my)     : {} seconds".format(detectImgEnd - detectImgStart))
#for (x, y, w, h) in rects:
#		cv2.rectangle(testImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
#cv2.imshow("Before NMS", testImg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




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
nlevels = 100000
defaultHOG = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                               derivAperture,winSigma,histogramNormType,
                               L2HysThreshold,gammaCorrection,nlevels)
defaultHOG.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
inputImg = cv2.imread("/home/gsy/Documents/myPedeatrainDetestration/mypicture/2.jpeg")
height, width = inputImg.shape[:2]
temp1 = cv2.resize(inputImg, (np.int16(0.4*width), np.int16(0.4*height)), interpolation = cv2.INTER_CUBIC)
detectImgStart = time.time()    #start record multie detection time
testImg = temp1.copy()


#(rects, weights) = defaultHOG.detectMultiScale(testImg, winStride = winStride, padding=(8, 8), scale=1.001, useMeanshiftGrouping = False)
#for (x, y, w, h) in rects:
#		cv2.rectangle(testImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
#cv2.imshow("Before resize", testImg)

height, width = testImg.shape[:2]
if height <= 128 or width <= 64:
    beta = 1/np.min([height/128, width/64])
    realTestImg = cv2.resize(testImg, (np.int16(beta*width), np.int16(beta*height)), interpolation = cv2.INTER_CUBIC)
(rects, weights) = defaultHOG.detectMultiScale(realTestImg, winStride = winStride, padding=(8, 8), scale=1.0001, useMeanshiftGrouping = False)
detectImgEnd = time.time()
print("Time of detection(default): {} seconds".format(detectImgEnd - detectImgStart))
for (x, y, w, h) in rects:
		cv2.rectangle(realTestImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("After resize", realTestImg)
cv2.waitKey(0)
cv2.destroyAllWindows()



#
#defaultHOG.winSize
#defaultHOG.blockSize
#defaultHOG.blockStride
#defaultHOG.cellSize
#defaultHOG.nbins
#defaultHOG.derivAperture
#defaultHOG.winSigma
#defaultHOG.histogramNormType
#defaultHOG.L2HysThreshold
#defaultHOG.gammaCorrection
#defaultHOG.nlevels
