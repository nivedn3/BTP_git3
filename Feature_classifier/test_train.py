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

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.Session()

pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_AREA)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]]})
    return faces


data = glob(os.path.join('/home/psycholearner/projects/DCGAN-tensorflow/data/celebA','*.jpg'))
data = sorted(data)[:1000]

label_file = open('list_attr_celeba.txt','rb')
labels = []
for lines in label_file:
	label = lines.split(' ')
	label = [i for i in label if i != '']
	labels.append(label)

labels = labels[2:1002]
labels = [int(i[21]) for i in labels]
labels = np.array(labels)
labels[labels < 0] = 0
print labels

imgs = []

for i in data:

	if i == '/home/psycholearner/projects/DCGAN-tensorflow/data/celebA/000003.jpg':
		i = '/home/psycholearner/projects/DCGAN-tensorflow/data/celebA/000001.jpg'
	if i == '/home/psycholearner/projects/DCGAN-tensorflow/data/celebA/000004.jpg':
		i = '/home/psycholearner/projects/DCGAN-tensorflow/data/celebA/000001.jpg'
	img = cv2.imread(i)
	print i
	faces = getFace(img)

	print faces[0]['rect']
	crop = img[faces[0]['rect'][1]:faces[0]['rect'][3],faces[0]['rect'][0]:faces[0]['rect'][2]]
	img = crop
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
num_classes = 1

model = mini_XCEPTION(input_shape, num_classes)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print labels.shape


model.fit(imgs_d,labels, batch_size = 10, epochs=10)



 
