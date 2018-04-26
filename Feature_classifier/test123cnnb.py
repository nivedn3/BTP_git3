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
data_train = sorted(data_raw)[0:170]
data_test = sorted(data_raw)[200000:202599]

label_file = open('list_attr_celeba.txt','r')
labels_train = []
labels_test = []
for lines in label_file:
	label = lines.split(' ')
	label = [i for i in label if i != '']
	labels_train.append(label)
	labels_test.append(label)

labels_train = labels_train[2:170002]
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



labels_test = labels_test[170002:202601]
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
	  #final_label_train.append([1,0])
	  #final_label_train.append([1,0])
	  #final_label_train.append([1,0])
	  #final_label_train.append([1,0])
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

	print(v)
	img = cv2.imread(v,cv2.IMREAD_GRAYSCALE)
	if labels_train_no_beard[i] == 0:
	  t = img.shape
	  crp = img[0:t[0]*4/5,0:t[1]*4/5]
	  crp2 = img[t[0]*1/5:t[0],t[1]*1/5:t[1]]
	  crp3 = img[0:t[0],0:t[1]*4/5]
	  crp4 = img[0:t[0]*4/5,0:t[1]]
	  crp5 = img[t[0]*1/5:t[0]*4/5,t[1]*1/5:t[1]*4/5]
	  flip = cv2.flip(src=img, flipCode=1)
	  #imgs_train.append(np.expand_dims(cv2.resize(crp,(150,150)),axis = 0).astype(np.float32))
      #imgs_train.append(np.expand_dims(cv2.resize(crp2,(150,150)),axis = 0).astype(np.float32))
      #imgs_train.append(np.expand_dims(cv2.resize(crp3,(150,150)),axis = 0).astype(np.float32))
      #imgs_train.append(np.expand_dims(cv2.resize(crp4,(150,150)),axis = 0).astype(np.float32))
      #imgs_train.append(np.expand_dims(cv2.resize(crp5,(150,150)),axis = 0).astype(np.float32))
	  imgs_train.append(np.expand_dims(cv2.resize(flip,(64,64)),axis = 0).astype(np.float32))
	  imgs_train.append(np.expand_dims(cv2.resize(img,(64,64)),axis = 0).astype(np.float32))

	else:

	  imgs_train.append(np.expand_dims(cv2.resize(img,(64,64)),axis = 0).astype(np.float32))

imgs_d_train = np.concatenate(imgs_train, axis=0).astype(np.float32)

imgs_d_train = imgs_d_train.reshape(imgs_d_train.shape[0],imgs_d_train.shape[1],imgs_d_train[2],1)

imgs_test = []
for i in data_test:
  img = cv2.imread(v,cv2.IMREAD_GRAYSCALE)
  imgs_test.append(np.expand_dims(cv2.resize(img,(64,64)),axis = 0).astype(np.float32))
imgs_d_test = np.concatenate(imgs_test, axis=0).astype(np.float32)
imgs_d_test = imgs_d_test.reshape(imgs_d_test.shape[0],imgs_d_test.shape[1],imgs_d_test.shape[2],1)

input_shape = (64, 64, 1)
num_classes = 2

model = mini_XCEPTION(input_shape, num_classes)



#model.fit(imgs_d_train,final_label_train, batch_size = 64, epochs=5)

regularization = l2(0.01)


model.load_weights('/home/ubuntu/face_classification/trained_models/gender_models/gender_mini_XCEPTION.22-0.96.hdf5')

model.layers.pop()
model.layers.pop()
model.layers.pop()
residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(model.layers[-1].output)
residual = BatchNormalization()(residual)

x = SeparableConv2D(256, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(model.layers[-1].output)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(256, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
x = GlobalAveragePooling2D()(x)

output = Activation('softmax',name='predictions')(x)

model2 = Model(model.input, output)

for layer in model.layers:
	layer.trainable = False

model2.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

t=time.time()
s = np.arange(final_label_train.shape[0])
final_label_train = final_label_train[s]
imgs_d_train = imgs_d_train[s]

class_weight = class_weight.compute_class_weight('balanced', np.unique(final_label_train), final_label_train)

hist = model.fit(imgs_d_train, final_label_train,class_weight=class_weight, batch_size=32, epochs=1, verbose=1, validation_data=(imgs_d_test, final_label_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = model.evaluate(imgs_d_test, final_label_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

file = open('/home/ubuntu/BTP_git3/genderpreb.txt','a')
file.write(str(train_loss))
file.write("\n\n\n\n\n\n")
file.write(str(val_loss))
file.write("\n\n\n\n\n\n")
file.write(str(train_acc))
file.write("\n\n\n\n\n\n")
file.write(str(val_acc))
file.write("\n\n\n\n\n\n")
file.close()
model.save_weights("/home/ubuntu/BTP_git3/weights/genderpreb/model.h5")
