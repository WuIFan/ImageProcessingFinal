#!/usr/bin/python
import tensorflow as tf
import numpy as np
import os, sys
import cv2
from PIL import Image

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
training_epochs = 5
batch_size = 1

DNN_input_dim = 128*128*36
hidden1_dim = 32
hidden2_dim = 32	#256
with tf.name_scope('InputLayer_to_HiddenLayer1'):
	# input layer to hidden layer 1
	w1 = tf.Variable(tf.random_normal([DNN_input_dim, hidden1_dim]),name='weight1')
	b1 = tf.Variable(tf.random_normal([hidden1_dim]),name='bias1')
	a1 = tf.nn.sigmoid(tf.add(tf.matmul(flatten,w1),b1))

	# add summary
	tf.summary.histogram("w1", w1)
	tf.summary.histogram("b1", b1)
	tf.summary.histogram("a1", a1)

with tf.name_scope('HiddenLayer1_to_HiddenLayer2'):
	# input layer to hidden layer 2
	w2 = tf.Variable(tf.random_normal([hidden1_dim, hidden2_dim]),name='weight2')
	b2 = tf.Variable(tf.random_normal([hidden2_dim]),name='bias2')
	a2 = tf.nn.sigmoid(tf.add(tf.matmul(a1,w2),b2))

	# add summary
	tf.summary.histogram("w2", w2)
	tf.summary.histogram("b2", b2)
	tf.summary.histogram("a2", a2)


with tf.name_scope('HiddenLayer2_to_OutputLayer'):
	# hidden layer 2 to output layer
	w3 = tf.Variable(tf.random_normal([hidden2_dim, output_dim]),name='weight3')
	b3 = tf.Variable(tf.random_normal([output_dim]),name='bias3')
	y_pred = tf.nn.sigmoid(tf.add(tf.matmul(a2,w3),b3))

	# add summary
	tf.summary.histogram("w3", w3)
	tf.summary.histogram("b3", b3)
	tf.summary.histogram("y_pred", y_pred)

losses = []
val_losses = []
#tf.reset_default_graph()
#saver = tf.train.import_meta_graph('model/model.meta.meta')

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,"model/model.meta")
    sess.run(tf.global_variables_initializer())
    all_vars = tf.trainable_variables()
    for v in all_vars:
    	print(v.name)
    testData = cv2.imread("../Data/data01/image/image0163.png",0)
    testLabel = cv2.imread("../Data/data01/label/image0163.png",0)
    print(testData.shape)
    print(testLabel.shape)
    testData = testData.reshape(1,512*512)/255
    testLabel = testLabel.reshape(1,512*512)/255
    #print(testData)
    print(testData.shape)
    print(testLabel.shape)
   	#cv2.imshow("s",testData)
    
    output = sess.run(y_pred, feed_dict={x: testData})
    output = output.reshape(512,512)*255
    print(y_pred)
    print (output)
    cv2.imshow("o",output)
    cv2.waitKey(0)
