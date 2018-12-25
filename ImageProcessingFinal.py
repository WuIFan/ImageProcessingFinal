#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os, sys
from PIL import Image

from io import BytesIO

def dataPreprocess():
	imageData = os.listdir("../Data/data01/image")
	labelData = os.listdir("../Data/data01/label")
	imageData.sort()
	labelData.sort()

	writer = tf.python_io.TFRecordWriter("data_train.tfrecords")
	for i in range(0,len(imageData)):
		img = Image.open("../Data/data01/image/" + imageData[i])
		#if (i < 5):
		#	img.show()
		img = img.resize((512,512))
		img_raw = img.tobytes()
		example = tf.train.Example(  
			features=tf.train.Features(feature={  
				"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),  
				'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
			}))
		writer.write(example.SerializeToString())
	writer.close()

def readAndDecode(filename):
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'label': tf.FixedLenFeature([], tf.int64),
			'img_raw': tf.FixedLenFeature([], tf.string)
		})
	img = tf.decode_raw(features['img_raw'],tf.uint8)
	img = tf.cast(img, tf.float32)
	img = tf.reshape(img,[512,512])
	label = tf.cast(features['label'],tf.int32)
	return img, label

if __name__ == '__main__':
	dataPreprocess()
	batch = readAndDecode("data_train.tfrecords")
	init_op = tf.initialize_all_variables()

	with tf.Session() as sess:
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)

		for i in range(100):
			example, lab = sess.run(batch)
			img = Image.fromarray(example)
			#img.save("g_"+str(i)+".png")
		img.show()
		coord.request_stop()
		coord.join(threads)
		sess.close()



