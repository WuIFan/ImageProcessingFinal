#!/usr/bin/python
import numpy as np
import os, sys
import cv2
from PIL import Image
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from final_ui import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setupUi(self)
		self.onBindingUI()

	# Write your code below
	# UI components are defined in hw1_ui.py, please take a look.
	# You can also open hw1.ui by qt-designer to check ui components.

	def onBindingUI(self):
		self.btn1.clicked.connect(self.on_btn1_click)
		self.btn2.clicked.connect(self.on_btn2_click)
		self.btn3.clicked.connect(self.on_btn3_click)

	def on_btn1_click(self):
		fname = QFileDialog.getOpenFileName(self, 'Open file')
		global datapath
		datapath = fname[0]
		pix = QPixmap(fname[0])
		self.label_9.setPixmap(pix)
		'''
		img = cv2.imread(datapath,0)
		image = QtGui.QImage(img, img.shape[1],img.shape[0], img.shape[1] * 3,QtGui.QImage.Format_RGB888)	
		print(type(image))
		pix.fromImage(image)
		print("?",type(pix))
		'''
		#cv2.waitKey(0)


	def on_btn2_click(self):
		fname = QFileDialog.getOpenFileName(self, 'Open file')
		global groundpath
		groundpath = fname[0]
		pix = QPixmap(fname[0])
		self.label_10.setPixmap(pix)

	def on_btn3_click(self):
		# cboxImgNum to access to the ui object
		result,ground,inter,dc = runOne(datapath,groundpath)
		pix = QPixmap("output.png")
		self.label_11.setPixmap(pix)
		self.label_2.setText(str(result))
		self.label_4.setText(str(ground))
		self.label_5.setText(str(inter))
		self.label_8.setText(str(dc))
		
		
		'''
		img = QtGui.QImage(image, image.shape[1],image.shape[0], image.shape[1] * 3,QtGui.QImage.Format_Indexed8)
		img = img.rgbSwapped()
		pix = QPixmap()
		pix.fromImage(img)
		'''
		
		'''
		height, width = testData.shape
		bytesPerLine = 3 * width
		qImg = QImage(testData.data, width, height, bytesPerLine, QImage.Format_RGB888)
		self.label_2.setPixmap(qImg)
		'''


def runAll():
	data = os.listdir("../Data")
	data.sort()
	#imageData = os.listdir("../Data/data01/image")
	#labelData = os.listdir("../Data/data01/label")
	#imageData.sort()
	#labelData.sort()

	dcList = []
	totalNum = 0
	for d in range(0,len(data)):
		if 'data' in data[d]:
			print ("read data:" + data[d])
			imageData = os.listdir("../Data/" + data[d] + "/image")
			labelData = os.listdir("../Data/" + data[d] + "/label")
			imageData.sort()
			labelData.sort()
			oneDcList = []
			for i in range(0,len(imageData)):
				img = cv2.imread("../Data/" + data[d] + "/image/" + imageData[i],0)
				lab = cv2.imread("../Data/" + data[d] + "/label/" + labelData[i],0)
				#print(imageData[i])
				th = findThreshold(img)
				ret,output = cv2.threshold(img,th,255,cv2.THRESH_BINARY)

				clone = output.copy()
				mask = np.zeros([512,512],dtype = clone.dtype)
				#disThreshold = 8000

				clone,myContours = findContours(clone,output)
				mask = makeMask(mask,myContours)

				result,ground,inter,dc = calResult(mask,lab)
				dcList.append(dc)
				oneDcList.append(dc)
				#if i ==0:
				#	break

			#print (oneDcList,dcList)
			avg = sum(oneDcList)/len(imageData)
			print (data[d],"avg:",avg)
			print (data[d],"stdDev:",stdDev(avg,oneDcList))
			totalNum = totalNum + len(imageData)
			#break
			
	print ("Number of picture:",totalNum)
	allAvg = sum(dcList)/totalNum
	print ("avg DC for all",allAvg)
	print ("avg stdDev for all",stdDev(allAvg,dcList))

def  stdDev(avg,dcList):
	sums = 0
	for dc in dcList:
		sums =  sums + np.square(avg - dc)
	ans = np.sqrt(sums / len(dcList))
	print (len(dcList))
	return ans

def makeMask(mask,cnts):
	color = [255, 255, 255]
	#cv2.fillPoly(mask, cnts, color=color)
	cv2.drawContours(mask, cnts, -1,color, -1)
	return mask

def findContours(clone,output):
	result,contours,hierarchy = cv2.findContours(output,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	i = 0
	count = 0
	maxArea = 0
	maxX = 0
	maxY = 0
	myContours = []
	while(i < len(contours)):
		num = len(contours[i])
		#print(num)
		if num > 100:
			#print("num:",num)
			area = cv2.contourArea(contours[i])
			#print("area:",area)
			if (area > maxArea):
				maxArea = area
				M = cv2.moments(contours[i])
				maxX = int(M["m10"] / M["m00"])
				maxY = int(M["m01"] / M["m00"])
			#cv2.drawContours(clone, [contours[i]], 0, (0, 255, 0), -1)
			myContours.append(contours[i])
			#mask = makeMask(mask,contours[i])
			#cv2.circle(clone, (centerX, centerY), 7, (0, 0, 255), -1)
			count = count + 1
		i = i + 1	
	#print (count)

	myContours = delByDistance(myContours,maxX,maxY)

	return clone,myContours

def delByDistance(myContours,maxX,maxY):
	newContours = []
	for cont in myContours: 
		M = cv2.moments(cont)
		if M["m00"] == 0:
			print(cont)
			continue
		centerX = int(M["m10"] / M["m00"])
		centerY = int(M["m01"] / M["m00"])
		distance = np.square(maxX - centerX) + np.square(maxY - centerY)
		if distance > disThreshold:
			continue
		newContours.append(cont)
		#print("X:",centerX,"Y:",centerY,"distance:",distance)
	return newContours

def findThreshold(testData):
	adj = 100
	height, width = testData.shape
	count = 10000
	while count > 7000:
		count = 0
		adj = adj + 5
		ret,temp = cv2.threshold(testData,adj,255,cv2.THRESH_BINARY)
		for i in range(height):
			for j in range(width):
				if temp[i ,j] == 255:
					count = count + 1	
		#print (count)
	#print(adj)
	return adj

def calResult(img1,img2):
	height, width = img1.shape
	result = 0
	ground = 0
	inter = 0
	dc = 0
	for i in range(height):
			for j in range(width):
				if img1[i ,j] == 255:
					result = result + 1
				if img2[i, j] == 255:
					ground = ground + 1
				if img1[i ,j] == 255 and img2[i, j] == 255:
					inter = inter + 1
	dc = 2 * inter / (result + ground)
	dc = format(dc, '.5f')
	dc = float(dc)
	print("result:",result,"ground:",ground,"intersection:",inter,"DC:",dc)
	return result,ground,inter,dc

def runOne(datapath,groundpath):
	#testData = cv2.imread("../Data/data01/image/image0180.png",0)
	#testLabel = cv2.imread("../Data/data01/label/image0180.png",0)
	testData = cv2.imread(datapath,0)
	testLabel = cv2.imread(groundpath,0)
	th = findThreshold(testData)
	ret,output = cv2.threshold(testData,th,255,cv2.THRESH_BINARY)

	clone = output.copy()
	#clone = cv2.cvtColor(clone,cv2.COLOR_GRAY2RGB)
	mask = np.zeros([512,512],dtype = clone.dtype)
	#mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

	clone,myContours = findContours(clone,output)
	mask = makeMask(mask,myContours)

	#cv2.imshow("mask",mask)
	#clone = cv2.bitwise_and(clone, mask)
	#cv2.imshow("clone",clone)
	#cv2.imshow("threshold",output)
	#cv2.imshow("ground",testLabel)

	result,ground,inter,dc = calResult(mask,testLabel)
	cv2.imwrite("output.png",mask)
	return result,ground,inter,dc
	#cv2.waitKey(0)
if __name__ == '__main__':
	datapath = ""
	groundpath = ""
	disThreshold = 8000


	### GUI ###
	'''
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
	'''
	#########
	runAll()
