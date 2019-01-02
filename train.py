#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os, sys
import cv2
from PIL import Image

from io import BytesIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
			if '07' not in data[d] and '08' not in data[d]:
				print ("read train data:" + data[d])
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
			else:
				print ("read test data:" + data[d])
				testImageData = os.listdir("../Data/" + data[d] + "/image")
				testLabelData = os.listdir("../Data/" + data[d] + "/label")
				testImageData.sort()
				testLabelData.sort()
				for i in range(0,len(testImageData)):
					img = cv2.imread("../Data/" + data[d] + "/image/" + testImageData[i],0)
					lab = cv2.imread("../Data/" + data[d] + "/label/" + testLabelData[i],0)
					testImageList.append(img)
					testLabelList.append(lab)

	print ("Number of training picture:",totalNum)
	print ("Number of testing picture:",len(testImageList))
	imageList = np.asarray(imageList)
	labelList = np.asarray(labelList)
	testImageList = np.asarray(testImageList)
	testLabelList = np.asarray(testLabelList)
	imageList = imageList.reshape(len(imageList),512*512)/255
	labelList = labelList.reshape(len(labelList),512*512)/255
	testImageList = testImageList.reshape(len(testImageList),512*512)/255
	testLabelList = testLabelList.reshape(len(testLabelList),512*512)/255
	print (imageList.shape)

	return imageList,labelList,testImageList,testLabelList

	'''
	print ("read...")
	for i in range(0,len(imageData)):
		#path = "/Users/Ivans/Documents/Studying/ImageProcessingFinal/Data/data01/image/" + imageData[i]
		#img = cv2.imread(path,0)
		img = cv2.imread("../Data/data01/image/" + imageData[i],0)
		#print("../Data/data01/image/"+imageData[i])
		#if (i < 5):
		#	img.show()
		#img = img.resize((512,512))
		print(img)
		#img = img.tobytes()
		imageList.append(img)
	imageList = np.asarray(imageList)
	print (imageList.shape)
	#imageList.reshape(100,512*512)
	imageList.reshape(len(imageData),512*512)
	#imageArray.reshape(len(imageData),512,512)
	print (len(imageData))
	'''

if __name__ == '__main__':
	imageList,labelList,testImageList,testLabelList = readFile()
	print (imageList.shape)

	input_dim = 512*512
	output_dim = 512*512
	
	x = tf.placeholder("float",[None, input_dim])
	y = tf.placeholder("float",[None, output_dim])
	x_image = tf.reshape(x, [-1, 512, 512, 1])

	with tf.name_scope('Convolution_layer1'):
	    W_conv1 = tf.Variable(tf.random_normal([3, 3, 1, 16]), name='Conv1_weight')
	    b_conv1 = tf.Variable(tf.random_normal([16]), name='Conv1_bias')
	    conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1))
	    tf.summary.histogram("conv1", conv1)

	with tf.name_scope('Max-pooling_layer1'):
	    max_pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	    tf.summary.histogram("max_pool1", max_pool1)

	with tf.name_scope('Convolution_layer2'):
	    W_conv2 = tf.Variable(tf.random_normal([3, 3, 16, 36]), name='Conv2_weight')
	    b_conv2 = tf.Variable(tf.random_normal([36]), name='Conv2_bias')
	    conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(max_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME'), b_conv2))
	    tf.summary.histogram("conv2", conv2)
	    
	with tf.name_scope('Max-pooling_layer2'):
	    max_pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	    tf.summary.histogram("max_pool2", max_pool2)

	with tf.name_scope('Flatten_layer'):
		flatten = tf.reshape(max_pool2, [-1, 128 * 128 * 36])

	learning_rate = 0.001
	training_epochs = 1
	batch_size = 1

	DNN_input_dim = 128*128*36
	hidden1_dim = 32
	hidden2_dim = 32	#256
	with tf.name_scope('InputLayer_to_HiddenLayer1'):
		# input layer to hidden layer 1
		w1 = tf.Variable(tf.random_normal([DNN_input_dim, hidden1_dim]),name='weight1')
		b1 = tf.Variable(tf.random_normal([hidden1_dim]),name='bias1')
		a1 = tf.nn.relu(tf.add(tf.matmul(flatten,w1),b1))

		# add summary
		tf.summary.histogram("w1", w1)
		tf.summary.histogram("b1", b1)
		tf.summary.histogram("a1", a1)

	with tf.name_scope('HiddenLayer1_to_HiddenLayer2'):
		# input layer to hidden layer 2
		w2 = tf.Variable(tf.random_normal([hidden1_dim, hidden2_dim]),name='weight2')
		b2 = tf.Variable(tf.random_normal([hidden2_dim]),name='bias2')
		a2 = tf.nn.relu(tf.add(tf.matmul(a1,w2),b2))

		# add summary
		tf.summary.histogram("w2", w2)
		tf.summary.histogram("b2", b2)
		tf.summary.histogram("a2", a2)


	with tf.name_scope('HiddenLayer2_to_OutputLayer'):
		# hidden layer 2 to output layer
		w3 = tf.Variable(tf.random_normal([hidden2_dim, output_dim]),name='weight3')
		b3 = tf.Variable(tf.random_normal([output_dim]),name='bias3')
		y_pred = tf.add(tf.matmul(a2,w3),b3)
		print(y_pred)

		# add summary
		tf.summary.histogram("w3", w3)
		tf.summary.histogram("b3", b3)
		tf.summary.histogram("y_pred", y_pred)

	with tf.name_scope('Loss'):
		#loss = tf.reduce_mean(tf.abs(tf.subtract(y, y_pred)))
		#tf.summary.scalar("loss", loss)
		smooth = 1.
		product = tf.multiply(y_pred, y)
		intersection = tf.reduce_sum(product)
		coefficient = (2. * intersection + smooth) / (tf.reduce_sum(y_pred) + tf.reduce_sum(y) + smooth)
		loss = 1 - coefficient
		tf.summary.scalar("loss", loss)

	with tf.name_scope('Accuracy'):
		accuracy = tf.reduce_mean(tf.abs(tf.subtract(y, y_pred)))
		tf.summary.scalar("accuracy", accuracy)

	with tf.name_scope('Optimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
		merged_summary = tf.summary.merge_all()

	losses = []
	val_losses = []
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    
	    # 初始化Variables
	    sess.run(tf.global_variables_initializer())
	    #writer = tf.summary.FileWriter("log_cnn/", graph=sess.graph)
	    global_step = 0
	    
	    for epoch in range(training_epochs):
	        #num_batch = int(mnist.train.num_examples/batch_size)
	        num_batch = 1
	        for i in range(num_batch):
	            #imageList, labelList = mnist.train.next_batch(batch_size)
	            #batch_x_validation, batch_y_validation = mnist.validation.next_batch(batch_size)
	            
	            # training by optimizer
	            print("optimizer...")
	            sess.run(optimizer, feed_dict={x: imageList, y: labelList})
	            
	            # get training/validation loss & acc
	            print("loss,acc...")
	            batch_loss = sess.run(loss, feed_dict={x: imageList, y: labelList})
	            batch_acc = sess.run(accuracy, feed_dict={x: imageList, y: labelList})
	            
	            # 紀錄每個batch的summary並加到writer中
	            #global_step += 1
	            #result = sess.run(merged_summary, feed_dict={x: imageList, y: labelList})
	            #writer.add_summary(result,global_step)
	            
	        losses.append(batch_loss)
	        #val_losses.append(batch_val_loss)
	        print("Epoch:", '%d' % (epoch+1), ", loss=", batch_loss, ", acc=", batch_acc)
	        
	    # Test Dataset
	    #print ("Test Accuracy:", sess.run(accuracy, feed_dict={x: testImageList, y: testLabelList}))
	    saver.save(sess,"model/model.meta")
