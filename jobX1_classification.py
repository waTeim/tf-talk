#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
from tensorflow.contrib import skflow

import cPickle as pickle

### Training data

# Downloads, unpacks and reads DBpedia dataset.
#dbpedia = skflow.datasets.load_dataset('dbpedia')
#X_train, y_train = pandas.DataFrame(dbpedia.train.data)[1], pandas.Series(dbpedia.train.target)
#X_test, y_test = pandas.DataFrame(dbpedia.test.data)[1], pandas.Series(dbpedia.test.target)

# Download dbpedia_csv.tar.gz from
# https://drive.google.com/folderview?id=0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JH...
# https://drive.google.com/folderview?id=0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
# Unpack: tar -xvf dbpedia_csv.tar.gz

x_train_save_path = 'jobX1/x_train.npy';
x_test_save_path = 'jobX1/x_test.npy';
y_train_save_path = 'jobX1/y_train.npy';
y_test_save_path = 'jobX1/y_test.npy';
vocab_save_path = 'jobX1/vocab.pkl';
MAX_DOCUMENT_LENGTH = 400

n_words = 29613

### Process vocabulary
if not os.path.exists(x_train_save_path) or not os.path.exists(x_test_save_path) or not os.path.exists(y_train_save_path) or not os.path.exists(y_test_save_path):
   print("initializing train and test");
   train = pandas.read_csv('jobX1/train.csv', header=None)
   X_train, y_train = train[1], train[0]
   test = pandas.read_csv('jobX1/test.csv', header=None)
   X_test, y_test = test[1], test[0]
   vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
   print("saving vocab");
   X_train = np.array(list(vocab_processor.fit_transform(X_train)))
   X_test = np.array(list(vocab_processor.transform(X_test)))
   n_words = len(vocab_processor.vocabulary_)
   with open(vocab_save_path,"wb") as output: pickle.dump(vocab_processor,output,pickle.HIGHEST_PROTOCOL)
   print("saving train and test");
   np.save(x_train_save_path,X_train)
   np.save(y_train_save_path,y_train)
   np.save(x_test_save_path,X_test)
   np.save(y_test_save_path,y_test);
else:
   print("using saved form");
   X_train = np.load(x_train_save_path)
   X_test = np.load(x_test_save_path)
   y_train = np.load(y_train_save_path)
   y_test = np.load(y_test_save_path)


print('Total words: %d' % n_words)

### Models
# 16,16,32 seems pretty good

EMBEDDING_SIZE = 32
N_FILTERS = 16
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
N_CLASSES=694


def cnn_model(X, y):
    """2 layer Convolutional network to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = skflow.ops.conv2d(word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # Add a RELU for non linearity.
        conv1 = tf.nn.relu(conv1)
        # Max pooling across output of Convlution+Relu.
        pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1], 
            strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = skflow.ops.conv2d(pool1, N_FILTERS, FILTER_SHAPE2,
            padding='VALID')
        # Max across each filter to get useful features for classification.
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    # Apply regular WX + B and classification.
    return skflow.models.logistic_regression(pool2, y)

classifier_config = skflow.RunConfig(gpu_memory_fraction=0.95)

model_path = '/home/jeffw/tf/models/jobX1'
if os.path.exists(model_path):
    classifier = skflow.TensorFlowEstimator.restore(model_path,config=classifier_config)
else:
    classifier = skflow.TensorFlowEstimator(model_fn=cnn_model, n_classes=N_CLASSES,
        steps=3000, optimizer='Adam', learning_rate=0.005, continue_training=True,config=classifier_config)


# Continuously train for 100 steps & predict on test set.
try:
   classifier.fit(X_train, y_train, logdir='/home/jeffw/tf/logs/jobX1')
   score = metrics.accuracy_score(y_test, classifier.predict(X_test))
   print('Accuracy: {0:f}'.format(score))
   classifier.save(model_path)
except KeyboardInterrupt:
   classifier.save(model_path)

