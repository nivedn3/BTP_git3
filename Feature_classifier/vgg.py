from keras.models import Sequential
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from glob import glob
import os
import cv2
from keras.layers import Input
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.models import Model
import time

from align import detect_face
from models.cnn import mini_XCEPTION
import numpy as np


ite = 0


print("*************************************")
print("iteration numeber",ite)
print("*************************************")

data_raw = glob(os.path.join('/home/psycholearner/projects/DCGAN-tensorflow/data/celebA','*.jpg'))
data_train = sorted(data_raw)[0:200000]
data_test = sorted(data_raw)[200000:202599]

label_file = open('list_attr_celeba.txt','r')
labels_train = []
labels_test = []
for lines in label_file:
	label = lines.split(' ')
	label = [i for i in label if i != '']
	labels_train.append(label)
	labels_test.append(label)

labels_train = labels_train[2:200002]
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

	if labels_train_no_beard[i] == 0:
	  final_label_train.append([0,1])
	  final_label_train.append([0,1])

	else:

	  final_label_train.append([1,0])


final_label_train = np.array(final_label_train)


final_label_test = []
for i,v in enumerate(labels_test_genders):

	if labels_test_no_beards[i] == 1:

	  final_label_test.append([1,0])

	else:

	  final_label_test.append([0,1])


final_label_test = np.array(final_label_test)

imgs_train = []

for i,v in enumerate(data_train):

	img = cv2.imread(v)
	if labels_train_no_beard[i] == 0: 
	  
	  flip = cv2.flip(src=img, flipCode=1)
	  imgs_train.append(np.expand_dims(cv2.resize(img,(224,224)),axis = 0).astype(np.float32))
	  imgs_train.append(np.expand_dims(cv2.resize(flip,(224,224)),axis = 0).astype(np.float32))

	else:

	  imgs_train.append(np.expand_dims(cv2.resize(img,(224,224)),axis = 0).astype(np.float32))

imgs_d_train = np.concatenate(imgs_train, axis=0).astype(np.float32)

imgs_test = []

for i in data_test:

  img = cv2.imread(v)
  imgs_test.append(np.expand_dims(cv2.resize(img,(224,224)),axis = 0).astype(np.float32))

imgs_d_test = np.concatenate(imgs_test, axis=0).astype(np.float32)

image_input = Input(shape=(224, 224, 3))

model = VGG19(weights='imagenet',include_top=False)
model.summary()
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
num_classes = 2
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

custom_resnet_model2.summary()

for layer in custom_resnet_model2.layers[:-6]:
	layer.trainable = False

custom_resnet_model2.layers[-1].trainable

custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
class_weight = {0:1,1:23}
hist = custom_resnet_model2.fit(imgs_d_train, final_label_train,class_weight = class_weight, batch_size=32, epochs=12, verbose=1, validation_data=(imgs_d_test, final_label_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model2.evaluate(imgs_d_test, final_label_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

file = open('/home/ubuntu/BTP_git3/vgg.txt','a')
file.write(train_loss)
file.write("\n\n\n\n\n\n")
file.write(val_loss)
file.write("\n\n\n\n\n\n")
file.write(train_acc)
file.write("\n\n\n\n\n\n")
file.write(val_acc)
file.write("\n\n\n\n\n\n")
file.close()
model.save_weights("/home/ubuntu/BTP_git3/weights/vgg/model.h5")
