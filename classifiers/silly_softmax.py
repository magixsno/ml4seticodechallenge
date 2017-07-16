from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import pandas as pd
import numpy
import sys
import os
import time
from datetime import datetime

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  print(str(datetime.now()))

  # Reads the labels
  index_file = pd.read_csv(os.path.join('data', 'public_list_basic_v2_26may_2017.csv'))
  classifications = {}
  for index, data in index_file.iterrows():
  	classifications[data['UUID'] + '.dat.png'] = data['SIGNAL_CLASSIFICATION']

  # Gives the labels an id
  classifications_to_num = {'narrowband': 0, 'narrowbanddrd': 1, 'squiggle': 2, 'noise': 3}
  print((int)(len(os.listdir('data_out/'))/25*3/4))
  # Flattens the images & fills out the labels
  # This image flattening takes the average of all the pixels and does not change
  # the size of the iamges.
  im_size = Image.open("data_out/" + os.listdir('data_out/')[0])
  tens_out = numpy.zeros(shape=((int)(len(os.listdir('data_out/'))/25), im_size.size[0]*im_size.size[1]))
  labels_out = numpy.zeros(shape=((int)(len(os.listdir('data_out/'))/25), 4))
  img_count = 0
  for file_name in os.listdir('data_out/'):
      if img_count >= (int)(len(os.listdir('data_out/'))/25):
          break
      im = Image.open("data_out/" + file_name)
      pix = im.load()
      pix_count = 0
      for x in range(0, im.size[0]):
          for y in range(0, im.size[1]):
              tens_out[img_count][pix_count] = ((pix[x,y][0] + pix[x,y][1] + pix[x,y][2] + pix[x,y][3])/4.0)/255.0
              pix_count += 1
      #print(tens_out)
      labels_out[img_count][classifications_to_num[classifications[file_name]]] = 1
      img_count += 1
  print("Completed building")
  img_train, img_test = numpy.split(tens_out, [(int)(len(os.listdir('data_out/'))/25*3/4)], axis=0)
  label_train, label_test = numpy.split(labels_out, [(int)(len(os.listdir('data_out/'))/25*3/4)], axis=0)

  print("Completed Assignment")

  # Create the model
  x = tf.placeholder(tf.float32, [None, im_size.size[0]*im_size.size[1]])
  W = tf.Variable(tf.zeros([im_size.size[0]*im_size.size[1], 4]))
  b = tf.Variable(tf.zeros([4]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 4])
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  print("Training...")
  # Train - Lol wtf are you training if you aren't sampling?
  # TODO: Add sampling
  for _ in range(1000):
    sess.run(train_step, feed_dict={x: img_train, y_: label_train})

  print("Testing...")
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: img_test,
                                      y_: label_test}))
  print(str(datetime.now()))
  print("Done.") # We're seeing in average of 40% accuracy with small data sets.

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
