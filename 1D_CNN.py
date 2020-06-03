#!/usr/bin/env python
# coding: utf-8

# # Getting the Envirnment Setup

# In[1]:


get_ipython().system('pip install tensorflow-gpu==1.14')


# In[2]:


# Importing all of the necessary libraries

import tensorflow as tf

import numpy as np

import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt

import os

import keras
from keras import models
from keras import layers
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

from scipy import fftpack

import sklearn.preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

from seaborn import heatmap


# # Defining Functions

# In[3]:


# Functions for statistics to judge the effectiveness of the models

def modelRecall(trueResult, predictedResult):
    "Calculates recall (true positives/all positives)"

    truePos = keras.backend.sum(keras.backend.round(keras.backend.clip(trueResult * predictedResult, 0, 1)))
    allPositives = keras.backend.sum(keras.backend.round(keras.backend.clip(trueResult, 0, 1)))
    return truePos/(allPositives + keras.backend.epsilon())

def modelPrecision(trueResult, predictedResult):
    "Calculates precision (true positives/predicted positives)"

    truePos = keras.backend.sum(keras.backend.round(keras.backend.clip(trueResult * predictedResult, 0, 1)))
    predPositives = keras.backend.sum(keras.backend.round(keras.backend.clip(predictedResult, 0, 1)))
    return truePos/(predPositives + keras.backend.epsilon())

def modelF1Score(trueResult, predictedResult):
    "Calculates F1-Score (2*((precision*recall)/(precision+recall)))"

    precisionResult = modelPrecision(trueResult, predictedResult)
    recallResult = modelRecall(trueResult, predictedResult)
    return 2 * ((precisionResult*recallResult)/(precisionResult+recallResult+keras.backend.epsilon()))

def modelAccuracy(model, test_data, test_label):
    score = model.evaluate(test_data, test_label, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return

# Function to plot training and validation, as well as training and validation accuracies

def plotResults(model_history):
    
    # Plot training & validation
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Epoch-Loss Plot')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()

    # Plot training & validation accuracy values
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    return

# Function for fourier transform

def fourierTransformer(trainData, minFreq = np.nan, maxFreq = np.nan, lengthOfRecording = np.nan, removeDCOffset = True):
  """
  Returns the fourier transform for a gived dataset. 
  :param trainData: Array of time domain datasets, with each row being a differet dataset. 
  :param minFreq: The lower bound of frequencies you want returned. Default = np.nan (no cutoff)
  :param maxFreq: The upper bound of frequencies you want returned. Default = np.nan (no cutoff)
  :param lengthOfRecording: Length of recording in seconds. Must be provided if minFreq or maxFreq are provided.
  :param removeDCOffset: Whether you want remove the 0 frequency data (DC Offset). Default = True
  """
  
  output = np.zeros(trainData.shape)

  for i in range(trainData.shape[0]):
    output[i, :] = fftpack.fft(trainData[i, :]) 
  
  if not np.isnan(maxFreq) or not np.isnan(minFreq):
    assert not np.isnan(lengthOfRecording), "lengthOfRecording must be defined if minFreq or maxFreq are defined"
    fftFrequencies = fftpack.fftfreq(trainData.shape[1] * 2, d = lengthOfRecording/trainData.shape[1])

  if np.isnan(minFreq):
    minLim = int(removeDCOffset)
  else:
    minLim = np.where(fftFrequencies == minFreq)[0][0]

  if np.isnan(maxFreq):
    maxLim = output.shape[1]//2
  else:
    maxLim = np.where(fftFrequencies == maxFreq)[0][0]

  return np.abs(output[:, minLim:maxLim])


# # Loading the Data Files

# In[4]:


'''
Loading the different data sets to be used with the models
These data sets have already been preprocessed and prepared for the models
Two different data sets are used
The first data set is the above-in data set, where the classifier is either above the brain or in the brain
This simplifies the data the models will use to make sure that they can properly classify a simple binary classification problem

The second data set which is the multi-region set uses all 22 of the different brain regions for classification
and represents the actual classification problem

Each of the different data sets has data for training, testing, and validation
Because of this, there is no need for a train_test_split
'''

train_data_above_in = np.load('first_5m_train_input_3s_lfp_above_in_2recordings.npy')
test_data_above_in = np.load('first_5m_test_input_3s_lfp_above_in_2recordings.npy')
val_data_above_in = np.load('first_5m_val_input_3s_lfp_above_in_2recordings.npy')
train_label_above_in = np.load('first_5m_train_labels_3s_lfp_above_in_2recordings.npy')
test_label_above_in = np.load('first_5m_test_labels_3s_lfp_above_in_2recordings.npy')
val_label_above_in = np.load('first_5m_val_labels_3s_lfp_above_in_2recordings.npy')

train_data_multi = np.load('first_5m_train_input_3s_lfp_multi_region_4recordings.npy')
test_data_multi = np.load('first_5m_test_input_3s_lfp_multi_region_4recordings.npy')
val_data_multi = np.load('first_5m_val_input_3s_lfp_multi_region_4recordings.npy')
train_label_multi = np.load('first_5m_train_labels_3s_lfp_multi_region_4recordings.npy')
test_label_multi = np.load('first_5m_test_labels_3s_lfp_multi_region_4recordings.npy')
val_label_multi = np.load('first_5m_val_labels_3s_lfp_multi_region_4recordings.npy')


# In[5]:


# Expanding the input data for the CNNs, which requires 3-dimensional data sets

print(train_data_multi.shape)
print(test_data_multi.shape)
print(val_data_multi.shape)

train_data_above_in = np.expand_dims(train_data_above_in, axis=2)
test_data_above_in = np.expand_dims(test_data_above_in, axis=2)
val_data_above_in = np.expand_dims(val_data_above_in, axis=2)

train_data_multi = np.expand_dims(train_data_multi, axis=2)
test_data_multi = np.expand_dims(test_data_multi, axis=2)
val_data_multi = np.expand_dims(val_data_multi, axis=2)

print(train_data_multi.shape)
print(test_data_multi.shape)
print(val_data_multi.shape)


# # Building and Testing the Neural Networks

# In[29]:


# Model 1: Initial model on the simple data set
opt = keras.optimizers.adam(lr=0.01)

CNN_above_in_1 = models.Sequential()
CNN_above_in_1.add(layers.Conv1D(filters=6, kernel_size=300, activation='relu', input_shape=(7500,1)))
CNN_above_in_1.add(layers.Conv1D(filters=4, kernel_size=4, activation='relu'))
CNN_above_in_1.add(layers.Dropout(0.5))
CNN_above_in_1.add(layers.MaxPooling1D(pool_size=2))
CNN_above_in_1.add(layers.Flatten())
CNN_above_in_1.add(layers.Dense(44, activation='relu'))
CNN_above_in_1.add(layers.Dense(22, activation='softmax'))
CNN_above_in_1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[26]:


'''
#To save the model weights to a hard drive

checkpoint_filepath = '/content/drive/My Drive/CSCI_5391/weights_best.hdf5'
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
'''


# In[30]:


CNN_above_in_1_history = CNN_above_in_1.fit(train_data_above_in, train_label_above_in, 
                   validation_data=(val_data_above_in ,val_label_above_in),
                   epochs=20, batch_size=256)


# In[31]:


plotResults(CNN_above_in_1_history)
modelAccuracy(CNN_above_in_1, test_data_above_in, test_label_above_in)


# As is to be expected for this type of simple binary classification data, the model quickly reaches 95.13% accuracy in the first epoch, and levels off around 95.36% accuracy. This simple experiment shows that this data can be classified by a 1D CNN. To further test this model, and other models, more experiments are performed only utilizing the multi-categorical data. This data has 22 distinct classifications.

# In[10]:


# Model 1 on the harder, multi-region data

CNN_multi_1 = models.Sequential()
CNN_multi_1.add(layers.Conv1D(filters=6, kernel_size=300, activation='relu', input_shape=(7500,1)))
CNN_multi_1.add(layers.Conv1D(filters=4, kernel_size=4, activation='relu'))
CNN_multi_1.add(layers.Dropout(0.5))
CNN_multi_1.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_1.add(layers.Flatten())
CNN_multi_1.add(layers.Dense(44, activation='relu'))
CNN_multi_1.add(layers.Dense(22, activation='softmax'))
CNN_multi_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[11]:


CNN_multi_1_history = CNN_multi_1.fit(train_data_multi, train_label_multi, 
                   validation_data=(val_data_multi ,val_label_multi),
                   epochs=20, batch_size=256)


# In[12]:


plotResults(CNN_multi_1_history)
modelAccuracy(CNN_multi_1, test_data_multi, test_label_multi)


# A model showing good results over 20 epochs, leveling off around 95% accuracy.

# In[13]:


'''
Model2: Another model for the above-in data
For the convolutional layers: more filters, but smaller kernel sizes.
As well as a larger intermediate dense layer
'''

CNN_above_in_2 = models.Sequential()
CNN_above_in_2.add(layers.Conv1D(filters=32, kernel_size=4, activation='relu', input_shape=(7500,1)))
CNN_above_in_2.add(layers.Conv1D(filters=32, kernel_size=4, activation='relu'))
CNN_above_in_2.add(layers.Dropout(0.5))
CNN_above_in_2.add(layers.MaxPooling1D(pool_size=2))
CNN_above_in_2.add(layers.Flatten())
CNN_above_in_2.add(layers.Dense(50, activation='relu'))
CNN_above_in_2.add(layers.Dense(22, activation='softmax'))
CNN_above_in_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[14]:


CNN_above_in_2_history = CNN_above_in_2.fit(train_data_above_in, train_label_above_in, 
                   validation_data=(val_data_above_in ,val_label_above_in),
                   epochs=20, batch_size=256)


# In[22]:


# Model 2 on the multi-region data set

opt = keras.optimizers.adam(lr=0.008)

CNN_multi_2 = models.Sequential()
CNN_multi_2.add(layers.Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(7500,1)))
CNN_multi_2.add(layers.Conv1D(filters=16, kernel_size=4, activation='relu'))
CNN_multi_2.add(layers.Dropout(0.5))
CNN_multi_2.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_2.add(layers.Flatten())
CNN_multi_2.add(layers.Dense(50, activation='relu'))
CNN_multi_2.add(layers.Dense(22, activation='softmax'))
CNN_multi_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[23]:


CNN_multi_2_history = CNN_multi_2.fit(train_data_multi, train_label_multi, 
                   validation_data=(val_data_multi ,test_label_multi),
                   epochs=20, batch_size=128)


# In[24]:


plotResults(CNN_multi_2_history)
modelAccuracy(CNN_multi_2, test_data_multi, test_label_multi)


# This model reaches a plateau of 14.78% accuracy. Slight changes were made to the model such as learning rate, number of epochs, and batch size but there was no change in accuracy. This points to the fact that the base architecture of the model is inefficient at learning the classification of the data.

# In[22]:


# Model 3
# Variation of Model 1
# Changing the filter and kernel sizes

CNN_multi_3 = models.Sequential()
CNN_multi_3.add(layers.Conv1D(filters=6, kernel_size=50, activation='relu', input_shape=(7500,1)))
CNN_multi_3.add(layers.Conv1D(filters=4, kernel_size=25, activation='relu'))
CNN_multi_3.add(layers.Dropout(0.5))
CNN_multi_3.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_3.add(layers.Flatten())
CNN_multi_3.add(layers.Dense(44, activation='relu'))
CNN_multi_3.add(layers.Dense(22, activation='softmax'))
CNN_multi_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[23]:


CNN_multi_3_history = CNN_multi_3.fit(train_data_multi, train_label_multi, 
                   validation_data=(val_data_multi ,test_label_multi),
                   epochs=20, batch_size=256)


# In[24]:


plotResults(CNN_multi_3_history)
modelAccuracy(CNN_multi_3, test_data_multi, test_label_multi)


# This model performs very well on the training data, and only slightly worse on the test data. The accuracy for the validation data points to the fact that the model may be over training on the training data, or that the validation data is skewed in some way. For instance this could be due to not enough validation samples, or that the validation data is over represented in a specific category.

# In[25]:


'''
Model 4
Similar to the above model but slightly refined
Uses a SGD optimizer where learning rate can be adjusted
'''

from keras.optimizers import SGD
opt = SGD(lr=0.10)

CNN_multi_4 = models.Sequential()
CNN_multi_4.add(layers.Conv1D(filters=32, kernel_size=16, activation='relu', input_shape=(7500,1)))
#Increasing kernal size doesnt affect the model's performance
CNN_multi_4.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_4.add(layers.Conv1D(filters=16, kernel_size=8, activation='relu'))
#Increasing this layer's kernal size doesn't affect the model's performance
CNN_multi_4.add(layers.Dropout(0.5))
CNN_multi_4.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_4.add(layers.Flatten())
CNN_multi_4.add(layers.Dense(50, activation='relu'))
CNN_multi_4.add(layers.Dense(22, activation='softmax'))
CNN_multi_4.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
#Using the SGD optimizer increased accuracy by about 3%


# In[26]:


CNN_multi_4_history = CNN_multi_4.fit(train_data_multi, train_label_multi, 
                   validation_data=(val_data_multi ,val_label_multi),
                   epochs=20, batch_size=128)


# In[27]:


plotResults(CNN_multi_4_history)
modelAccuracy(CNN_multi_4, test_data_multi, test_label_multi)


# This model also shows good results on all three different types of data. It reaches a very high accuracy quickly, but levels off at that point. Because of this behavior, this model may be inconsistent.

# In[28]:


'''
Model 5
Uses the same layers as model 4, but with an adagrad optimizer
Only 2 dense layers but with a larger filter and kernal size in convolution layers
'''

CNN_multi_5 = models.Sequential()
CNN_multi_5.add(layers.Conv1D(filters=32, kernel_size=16, activation='relu', input_shape=(7500,1)))
CNN_multi_5.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_5.add(layers.Conv1D(filters=16, kernel_size=8, activation='relu'))
CNN_multi_5.add(layers.Dropout(0.5))
CNN_multi_5.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_5.add(layers.Flatten())
CNN_multi_5.add(layers.Dense(50, activation='relu'))
CNN_multi_5.add(layers.Dense(22, activation='softmax'))
CNN_multi_5.compile(loss='categorical_crossentropy', optimizer= 'adagrad', metrics=['accuracy'])


# In[29]:


CNN_multi_5_history = CNN_multi_5.fit(train_data_multi, train_label_multi, 
                   validation_data=(val_data_multi, val_label_multi),
                   epochs=20, batch_size=128)


# In[30]:


plotResults(CNN_multi_5_history)
modelAccuracy(CNN_multi_5, test_data_multi, test_label_multi)


# Another great model that performs well on all three data sets. 

# In[24]:


'''
Model 6
Further experimentation
Smaller filter and kernel sizes on the 1D convolutional layers
More dense layers of smaller sizes

'''

adagrad = keras.optimizers.Adagrad(lr=0.01)

CNN_multi_6 = models.Sequential()
CNN_multi_6.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu', input_shape=(7500,1)))
CNN_multi_6.add(layers.MaxPooling1D(pool_size=4))
CNN_multi_6.add(layers.Conv1D(filters=16, kernel_size=4, activation='relu'))
CNN_multi_6.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_6.add(layers.Dropout(0.5))
CNN_multi_6.add(layers.Conv1D(filters=8, kernel_size=2, activation='relu'))
CNN_multi_6.add(layers.MaxPooling1D(pool_size=2))
CNN_multi_6.add(layers.Flatten())
CNN_multi_6.add(layers.Dense(32, activation='relu'))
CNN_multi_6.add(layers.Dense(16, activation='relu'))
CNN_multi_6.add(layers.Dense(8, activation='relu'))
CNN_multi_6.add(layers.Dense(4, activation='relu'))
CNN_multi_6.add(layers.Dense(22, activation='softmax'))
CNN_multi_6.compile(loss='categorical_crossentropy', optimizer= 'adagrad', metrics=['accuracy'])


# In[25]:


CNN_multi_6_history = CNN_multi_6.fit(train_data_multi, train_label_multi, 
                   validation_data=(val_data_multi , val_label_multi),
                   epochs=20, batch_size=128)


# In[26]:


plotResults(CNN_multi_6_history)
modelAccuracy(CNN_multi_6, test_data_multi, test_label_multi)


# This model reaches a similar point as the other failed models and is unable to get above around 14.5% accuracy.

# In[ ]:


'''
#Code for saving the best model to a hard drive and printing out the more indepth and involved metrics

CNN_multi_6.summary()

keras.utils.plot_model(CNN_multi_7, show_layer_names=False, show_shapes=True, to_file='')

model_data = CNN_multi_7.to_json()

with open('filepath/model.json', 'w') as outputFile:
  outputFile.write(model_data)
  
CNN_multi_7.save_weights('filepath/modelWeight.h5')

with open('filepath/trainHistoryDict', 'wb') as file_pi:
  pickle.dump(CNN_multi_7_history.history, file_pi)
  
model_history = CNN_multi_7_history.history

# Loading metrics from the model
with open('filepath/trainHistoryDict', 'rb') as f:
  new_model_history = pickle.load(f)
  
# Printing final numbers
model_metrics = new_model_history.keys()
for i in modelMetrics:
  print(i + ": {}".format(new_model_history[i][-1]))
  
plotResults(CNN_multi_7_history)
modelAccuracy(CNN_multi_7, test_data_multi, test_label_multi)
'''


# This model combines a Long Short-Term Memory (LSTM) layer with the 1D CNN. It is unable to reach an adequate accuracy and quickly levels off. 

# In[19]:


'''
Model 7
This is a model that combines CNN layers with LSTM layers
'''

from keras.optimizers import SGD
opt = SGD(lr=0.10)

LSTM_multi_1 = models.Sequential()
LSTM_multi_1.add(layers.Conv1D(filters=32, kernel_size=16, activation='relu', input_shape=(7500,1)))
LSTM_multi_1.add(layers.MaxPooling1D(pool_size=2))
LSTM_multi_1.add(layers.Conv1D(filters=16, kernel_size=8, activation='relu'))
LSTM_multi_1.add(layers.MaxPooling1D(pool_size=2))
LSTM_multi_1.add(LSTM(10))
LSTM_multi_1.add(layers.Dense(22, activation='softmax'))
LSTM_multi_1.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])


# In[22]:


LSTM_multi_1_history = LSTM_multi_1.fit(train_data_multi, train_label_multi, 
                   validation_data=(val_data_multi ,val_label_multi),
                   epochs=20, batch_size=256)


# In[23]:


plotResults(LSTM_multi_1_history)
modelAccuracy(LSTM_multi_1, test_data_multi, test_label_multi)


# In[ ]:




