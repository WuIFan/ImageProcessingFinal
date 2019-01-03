#!/usr/bin/python
import numpy as np
import os, sys
import cv2
from PIL import Image

def readFile():
	data = os.listdir("../Data")
	data.sort()
	#imageData = os.listdir("../Data/data01/image")
	#labelData = os.listdir("../Data/data01/label")
	#imageData.sort()
	#labelData.sort()

	imageList = []
	labelList = []
	testImageList = []
	testLabelList = []
	totalNum = 0
	for d in range(0,len(data)):
		if 'data' in data[d]:
			print ("read data:" + data[d])
			imageData = os.listdir("../Data/" + data[d] + "/image")
			labelData = os.listdir("../Data/" + data[d] + "/label")
			imageData.sort()
			labelData.sort()
			for i in range(0,len(imageData)):
				img = cv2.imread("../Data/" + data[d] + "/image/" + imageData[i],0)
				lab = cv2.imread("../Data/" + data[d] + "/label/" + labelData[i],0)
				imageList.append(img)
				labelList.append(lab)

			totalNum = totalNum + len(imageData)
			break
			

	print ("Number of training picture:",totalNum)
	return imageList,labelList

if __name__ == '__main__':
	imageList,labelList = readFile()

	'''
	count = 0
	for label in labelList:
		label = label.reshape(1,512*512)
		for pix in label:
			for p in pix:
				if p != 0:
					count = count + 1
	print (count/len(labelList))
	'''

	testData = cv2.imread("../Data/data01/image/image0163.png",0)
	testLabel = cv2.imread("../Data/data01/label/image0163.png",0)
	ret,output = cv2.threshold(testData,180,255,cv2.THRESH_BINARY)
	cv2.imshow("o",output)
	cv2.imshow("ground",testLabel)
	cv2.waitKey(0)