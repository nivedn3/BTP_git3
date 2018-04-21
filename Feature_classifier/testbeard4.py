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



for ite in range(20):

  print("*************************************")
  print("iteration numeber",ite)
  print("*************************************")
  

  data_raw = glob(os.path.join('/home/psycholearner/projects/DCGAN-tensorflow/data/celebA','*.jpg'))
  data_train = sorted(data_raw)[ite*100:100 + ite*100]
  data_test = sorted(data_raw)[200000:2000599]


  label_file = open('list_attr_celeba.txt','r')
  labels_train = []
  labels_test = []
  for lines in label_file:
    label = lines.split(' ')
    label = [i for i in label if i != '']
    labels_train.append(label)
    labels_test.append(label)

  labels_train = labels_train[2 + ite*100:102 + ite*100]
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



  labels_test = labels_test[200002:200601]
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
    print v
    if labels_train_no_beard[i] == 0: 
      
      flip = cv2.flip(src=img, flipCode=1)
      imgs_train.append(np.expand_dims(cv2.resize(img,(150,150)),axis = 0).astype(np.float32))
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

  file = open('/home/ubuntu/BTP_git3/logbeard4.txt','a')

  model = mini_XCEPTION(input_shape, num_classes)
  if ite:
    model.load_weights("/home/ubuntu/BTP_git3/weights/testbeard4/"+"model_%d.h5"%(ite-1))
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  #s = np.arange(final_label_train.shape[0])
  #final_label_train = final_label_train[s]
  #imgs_d_train = imgs_d_train[s]

  model.fit(imgs_d_train,final_label_train, batch_size = 64, epochs=5)

  test_acc = model.evaluate(imgs_d_test,final_label_test)
  print(test_acc)
  file.write("iter"+ str(ite) + "\n")
  file.write("acc"+str(test_acc)+ "\n")
  file.write("*****************"+ "\n")
  model.save_weights("/home/ubuntu/BTP_git3/weights/testbeard4/"+"model_%d.h5"%ite)
