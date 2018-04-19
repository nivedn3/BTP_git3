from keras.models import Sequential
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from glob import glob
import os
import cv2
from align import detect_face
from models.cnn import mini_XCEPTION
import numpy as np




data = glob(os.path.join('/home/psycholearner/projects/DCGAN-tensorflow/data/celebA','*.jpg'))
data = sorted(data)[:1000]

label_file = open('list_attr_celeba.txt','rb')
labels = []
for lines in label_file:
	label = lines.split(' ')
	label = [i for i in label if i != '']
	labels.append(label)

labels = labels[2:1002]
labels = [int(i[23]) for i in labels]
labels = np.array(labels)
labels[labels < 0] = 0
print labels
test = []
for i in labels:

    if i == 1:

        test.append([1,0])

    else:

        test.append([0,1])


print "************************************************************"
test = np.array(test)
print test.shape
imgs = []

for i in data:

	img = cv2.imread(i)
	imgs.append(np.expand_dims(cv2.resize(img,(150,150)),axis = 0).astype(np.float32))

imgs_d = np.concatenate(imgs, axis=0).astype(np.float32)
print imgs_d[0].shape

print "test"





'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''
input_shape = (150,150,3)
num_classes = 2

model = mini_XCEPTION(input_shape, num_classes)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print labels.shape


model.fit(imgs_d,test, batch_size = 10, epochs=10)



 
