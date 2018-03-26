import cv2
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from random import shuffle

lines = []

# READING DATA

with open('./data_leo/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	for line in reader:
		lines.append(line)

# DATA WITHOUT GENERATOR

#images = []
#measurements = []
#
#for line in lines:
#	steering_center = float(line[3])
#	
#	correction = 0.2
#	steering_left = steering_center + correction
#	steering_right = steering_center - correction
#
#	path = './data_leo/IMG/'
#	img_center = cv2.imread(path + line[0].split('/')[-1])
#	img_left = cv2.imread(path + line[1].split('/')[-1])
#	img_right = cv2.imread(path + line[2].split('/')[-1])
#	
#	images.append(img_center)
#	images.append(img_left)
#	images.append(img_right)
#
#	measurements.append(steering_center)
#	measurements.append(steering_left)
#	measurements.append(steering_right)
#
#X_train = np.array(images)
#y_train = np.array(measurements)


# SPLIT DATA

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# ADDING BRIGHTNESS AND DARKNESS 

#def addBrightness(imagen):
#	imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
#	h, s, v = cv2.split(imagen)
#	v[v < 235] += 20
#	imagen = cv2.merge((h,s,v))
#	imagen = cv2.cvtColor(imagen, cv2.COLOR_HSV2RGB)
#	return imagen
#
#def addDarkness(imagen):
#	imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
#	h, s, v = cv2.split(imagen)
#	v[v > 20] -= 20
#	imagen = cv2.merge((h,s,v))
#	imagen = cv2.cvtColor(imagen, cv2.COLOR_HSV2RGB)
#	return imagen

# USING GENERATOR

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while True:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			angles = []
			for i, batch_sample in enumerate(batch_samples):
				path = './data_leo/IMG/'
				img_center = cv2.imread(path + batch_sample[0].split('/')[-1])
				img_left = cv2.imread(path + batch_sample[1].split('/')[-1])
				img_right = cv2.imread(path + batch_sample[2].split('/')[-1])
				
				img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
				img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
				img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
				
				images.append(img_center)
				images.append(img_left)
				images.append(img_right)
			
				steering_center = float(batch_sample[3])				
				correction = 0.2

				steering_left = steering_center + correction
				steering_right = steering_center - correction

				angles.append(steering_center)
				angles.append(steering_left)
				angles.append(steering_right)

#				if i % 3 == 0:
#					images.append(addBrightness(img_center))
#					angles.append(steering_center)
#					images.append(addDarkness(img_center))
#					angles.append(steering_center)
#				elif i % 3 == 1:				
#					images.append(addBrightness(img_left))
#					angles.append(steering_left)
#					images.append(addDarkness(img_left))
#					angles.append(steering_left)		
#				else:		
#					images.append(addBrightness(img_right))
#					angles.append(steering_right)
#					images.append(addDarkness(img_right))
#					angles.append(steering_right)		
#
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)
			

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# LeNet
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D()) # default poolsize(2,2) stride(2,2) 
# model.add(Flatten()) 
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# AlexNet
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))

# model.add(Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Convolution2D(256, 5, 5, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Convolution2D(384, 3, 3, padding='same', activation='relu'))
# model.add(Convolution2D(384, 3, 3, padding='same', activation='relu'))
# model.add(Convolution2D(256, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Flatten())
# model.add(Dense(2048))
# model.add(Dense(2048))
# model.add(Dense(1))

# ZFNet
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))

#model.add(Convolution2D(64, 7, 7, activation='relu'))

#... falta


# VGG
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))

# model.add(Convolution2D(64, 3, 3, padding='same', activation='relu'))
# model.add(Convolution2D(64, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Convolution2D(128, 3, 3, padding='same', activation='relu'))
# model.add(Convolution2D(128, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Convolution2D(256, 3, 3, padding='same', activation='relu'))
# model.add(Convolution2D(256, 3, 3, padding='same', activation='relu'))
# model.add(Convolution2D(256, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Convolution2D(512, 3, 3, padding='same', activation='relu'))
# model.add(Convolution2D(512, 3, 3, padding='same', activation='relu'))
# model.add(Convolution2D(512, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Flatten()) 
# model.add(Dense(4096))
# model.add(Dense(4096))
# model.add(Dense(1))



# NVIDIA
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
# model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
# model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
# model.add(Convolution2D(64,3,3,activation='relu'))
# model.add(Convolution2D(64,3,3,activation='relu'))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))


# NVIDIA Net improved
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
# model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D(64,3,3,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution2D(64,3,3,activation='relu'))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

# FIT MODEL

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=50)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data = validation_generator, nb_val_samples=len(validation_samples),nb_epoch=100, verbose=1)

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epochs')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
