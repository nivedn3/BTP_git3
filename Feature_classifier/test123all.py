from keras.models import Sequential
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from glob import glob
import os
import cv2
from keras.layers import Input
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.models import Model
import time
from align import detect_face
from models.cnn import *
import numpy as np
from sklearn.utils import class_weight




ite = 0

print("*************************************")
print("iteration numeber",ite)
print("*************************************")

data_raw = glob(os.path.join('/home/psycholearner/projects/DCGAN-tensorflow/data/celebA','*.jpg'))
data_train = sorted(data_raw)[0:200]
data_test = sorted(data_raw)[200000:202599]

label_file = open('list_attr_celeba.txt','r')
labels_train = []
labels_test = []
for lines in label_file:
	label = lines.split(' ')
	label = [i for i in label if i != '']
	labels_train.append(label)
	labels_test.append(label)

labels_train = labels_train[2:202]
labels_train_gender = [int(i[21]) for i in labels_train]
labels_train_moustache = [int(i[23]) for i in labels_train]
labels_train_glass = [int(i[16]) for i in labels_train]
labels_train_no_beard = [int(i[25]) for i in labels_train]



labels_train_gender = np.array(labels_train_gender)
labels_train_glass = np.array(labels_train_glass)
labels_train_no_beard = np.array(labels_train_no_beard)
labels_train_moustache = np.array(labels_train_moustache)
labels_train_gender[labels_train_gender < 0] = 0
labels_train_moustache[labels_train_moustache < 0] = 0
labels_train_glass[labels_train_glass < 0] = 0
labels_train_no_beard[labels_train_no_beard < 0] = 0



labels_test = labels_test[200002:202601]
labels_test_genders = [int(i[21]) for i in labels_test]
labels_test_moustaches = [int(i[23]) for i in labels_test]
labels_test_glasss = [int(i[16]) for i in labels_test]
labels_test_no_beards = [int(i[25]) for i in labels_test]



#labels = np.array(labels)
labels_test_genders = np.array(labels_test_genders)
labels_test_glasss = np.array(labels_test_glasss)
labels_test_no_beards = np.array(labels_test_no_beards)
labels_test_moustaches = np.array(labels_test_moustaches)
labels_test_genders[labels_test_genders < 0] = 0
labels_test_moustaches[labels_test_moustaches < 0] = 0
labels_test_glasss[labels_test_glasss < 0] = 0
labels_test_no_beards[labels_test_no_beards < 0] = 0

final_label_train = []
  for i,v in enumerate(labels_train_gender):

  	for i in range(7):
      final_label_train.append([labels_train_gender[i],labels_train_moustache[i],labels_train_no_beard[i]])

  final_label_train = np.array(final_label_train)


  final_label_test = []
  for i,v in enumerate(labels_test_genders):

  	for i in range(7):
      final_label_test.append([labels_test_genders[i],labels_test_moustaches[i],labels_test_no_beards[i]])

  final_label_test = np.array(final_label_test)


'''
final_label_train = []
for i,v in enumerate(labels_train_gender):

	if labels_train_moustache[i] == 1:

	  final_label_train.append([1,0])
	  final_label_train.append([1,0])
	  final_label_train.append([1,0])
	  final_label_train.append([1,0])
	  final_label_train.append([1,0])
	  final_label_train.append([1,0])
	else:

	  final_label_train.append([0,1])


final_label_train = np.array(final_label_train)


final_label_test = []
for i,v in enumerate(labels_test_genders):

	if labels_test_moustaches[i] == 1:

	  final_label_test.append([1,0])

	else:

	  final_label_test.append([0,1])
'''

final_label_test = np.array(final_label_test)

imgs_train = []

for i in data_train:
	img = cv2.imread(i)
	t = img.shape
	crp = img[0:t[0]*4/5,0:t[1]*4/5]
	crp2 = img[t[0]*1/5:t[0],t[1]*1/5:t[1]]
	crp3 = img[0:t[0],0:t[1]*4/5]
	crp4 = img[0:t[0]*4/5,0:t[1]]
	crp5 = img[t[0]*1/5:t[0]*4/5,t[1]*1/5:t[1]*4/5]
	flip = cv2.flip(src=img, flipCode=1)
	imgs_train.append(np.expand_dims(cv2.resize(crp,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(crp2,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(crp3,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(crp4,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(crp5,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(flip,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(img,(150,150)),axis = 0).astype(np.float32))
imgs_d_train = np.concatenate(imgs_train, axis=0).astype(np.float32)


imgs_test = []
for i in data_test:
	img = cv2.imread(i)
	t = img.shape
	crp = img[0:t[0]*4/5,0:t[1]*4/5]
	crp2 = img[t[0]*1/5:t[0],t[1]*1/5:t[1]]
	crp3 = img[0:t[0],0:t[1]*4/5]
	crp4 = img[0:t[0]*4/5,0:t[1]]
	crp5 = img[t[0]*1/5:t[0]*4/5,t[1]*1/5:t[1]*4/5]
	flip = cv2.flip(src=img, flipCode=1)
	imgs_train.append(np.expand_dims(cv2.resize(crp,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(crp2,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(crp3,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(crp4,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(crp5,(150,150)),axis = 0).astype(np.float32))
	imgs_train.append(np.expand_dims(cv2.resize(flip,(150,150)),axis = 0).astype(np.float32))
	imgs_test.append(np.expand_dims(cv2.resize(img,(150,150)),axis = 0).astype(np.float32))
imgs_d_test = np.concatenate(imgs_test, axis=0).astype(np.float32)

'''

for i,v in enumerate(data_train):

	print(v)
	img = cv2.imread(v)
	if labels_train_moustache[i] == 1: 
	  t = img.shape
	  crp = img[0:t[0]*4/5,0:t[1]*4/5]
	  crp2 = img[t[0]*1/5:t[0],t[1]*1/5:t[1]]
	  crp3 = img[0:t[0],0:t[1]*4/5]
	  crp4 = img[0:t[0]*4/5,0:t[1]]
	  crp5 = img[t[0]*1/5:t[0]*4/5,t[1]*1/5:t[1]*4/5]
	  flip = cv2.flip(src=img, flipCode=1)
	  imgs_train.append(np.expand_dims(cv2.resize(crp,(150,150)),axis = 0).astype(np.float32))
      imgs_train.append(np.expand_dims(cv2.resize(crp2,(150,150)),axis = 0).astype(np.float32))
      imgs_train.append(np.expand_dims(cv2.resize(crp3,(150,150)),axis = 0).astype(np.float32))
      imgs_train.append(np.expand_dims(cv2.resize(crp4,(150,150)),axis = 0).astype(np.float32))
      imgs_train.append(np.expand_dims(cv2.resize(crp5,(150,150)),axis = 0).astype(np.float32))
	  imgs_train.append(np.expand_dims(cv2.resize(flip,(150,150)),axis = 0).astype(np.float32))

	else:

	  imgs_train.append(np.expand_dims(cv2.resize(img,(150,150)),axis = 0).astype(np.float32))

imgs_d_train = np.concatenate(imgs_train, axis=0).astype(np.float32)

imgs_test = []

for i in data_test:

  img = cv2.imread(v)
  imgs_test.append(np.expand_dims(cv2.resize(img,(150,150)),axis = 0).astype(np.float32))

imgs_d_test = np.concatenate(imgs_test, axis=0).astype(np.float32)
'''
input_shape = (150,150,3)
num_classes = 3


model = tiny_XCEPTION(input_shape, num_classes)

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

#model.fit(imgs_d_train,final_label_train, batch_size = 64, epochs=5)

t=time.time()
s = np.arange(final_label_train.shape[0])
final_label_train = final_label_train[s]
imgs_d_train = imgs_d_train[s]

class_weight = class_weight.compute_class_weight('balanced', np.unique(final_label_train), final_label_train)




hist = model.fit(imgs_d_train, final_label_train, batch_size=32, epochs=1, verbose=1, validation_data=(imgs_d_test, final_label_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = model.evaluate(imgs_d_test, final_label_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

file = open('/home/ubuntu/BTP_git3/test123all.txt','a')
file.write(str(train_loss))
file.write("\n\n\n\n\n\n")
file.write(str(val_loss))
file.write("\n\n\n\n\n\n")
file.write(str(train_acc))
file.write("\n\n\n\n\n\n")
file.write(str(val_acc))
file.write("\n\n\n\n\n\n")
file.close()
model.save_weights("/home/ubuntu/BTP_git3/weights/test123all/model.h5")
