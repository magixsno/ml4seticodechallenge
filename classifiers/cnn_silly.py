from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import os
tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  sess = tf.Session()
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  #print(features)
  input_layer = tf.reshape(features, [-1, 40, 40, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 300, 300, 3]
  # Output Tensor Shape: [batch_size, 300, 300, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 300, 300, 32]
  # Output Tensor Shape: [batch_size, 300, 300, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 300, 300, 32]
  # Output Tensor Shape: [batch_size, 150, 150, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 150, 150, 64]
  # Output Tensor Shape: [batch_size, 75, 75, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 75, 75, 64]
  # Output Tensor Shape: [batch_size, 75 * 75 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 10 * 10 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 75 * 75 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 7]
  logits = tf.layers.dense(inputs=dropout, units=7)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=7)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def shared_input():
    # Reads the labels
    index_file = pd.read_csv(os.path.join('data', 'public_list_primary_v3_full_21june_2017.csv'))
    classifications = {}
    for index, data in index_file.iterrows():
      classifications[data['UUID'] + '.png'] = data['SIGNAL_CLASSIFICATION']

    # Gives the labels an id
    classifications_to_num = {'narrowband': 0, 'narrowbanddrd': 1, 'squiggle': 2, 'noise': 3, 'squigglesquarepulsednarrowband': 4, 'squarepulsednarrowband': 6, 'brightpixel': 7}
    dirname = 'primary_small/data_out/primary_small_v3/'

    # Flattens the images & fills out the labels
    # This image flattening takes the average of all the pixels and does not change
    # the size of the iamges.
    num_labels = 7
    image_list = [tf.image.decode_png(tf.read_file(dirname + filename), channels=3) for filename in os.listdir(dirname)]
    image_list_out = [tf.image.resize_images(image, [40, 40]) for image in image_list]
    labels_out = [classifications_to_num[classifications[filename]] for filename in os.listdir(dirname)]
    #print(labels_out)
    tens_out = tf.stack(image_list_out)
    print("Completed building")

    img_train = tf.split(tens_out, 11, axis=0)
    label_train, label_test = np.split(labels_out, [90], axis=0)

    # Load training and eval data
    train_data = tf.concat(img_train[:10], axis=0)
    train_labels = tf.convert_to_tensor(np.asarray(label_train, dtype=np.int32))
    eval_data = img_train[10]
    eval_labels = tf.convert_to_tensor(np.asarray(label_test, dtype=np.int32))
    print(train_data.get_shape())
    print(train_labels.get_shape())
    print(eval_data.get_shape())
    print(eval_labels.get_shape())
    print("Completed Assignment")
    return train_data, train_labels, eval_data, label_test

def my_input_fn():
    train_data, train_labels, eval_data, eval_labels = shared_input()
    return train_data, train_labels

def my_eval_fn():
    train_data, train_labels, eval_data, label_test = shared_input()
    with tf.Session():
        return eval_data.eval(), np.asarray(label_test, dtype=np.int32)

def main(unused_argv):
  print(str(datetime.now()))
  eval_data, eval_labels = my_eval_fn()

  #val, val2 = my_input_fn()

  # Create the Estimator
  special_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/seticode_convnet_model2")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=10)

  # Train the model
  special_classifier.fit(
      input_fn=my_input_fn,
      steps=20,
      monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }
  print("Evaluating...")
  # Evaluate the model and print results
  eval_results = special_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)
  print("Completed Eval.")


if __name__ == "__main__":
  tf.app.run()
