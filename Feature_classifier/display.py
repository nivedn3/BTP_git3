import cv2
from glob import glob
import os
import numpy as np

ite = 0
data_raw = glob(os.path.join('/home//psycholearner/projects/DCGAN-tensorflow/data/celebA','*.jpg'))
data_train = sorted(data_raw)[ite*10000:10000 + ite*10000]

img = cv2.imread(data_train[313])


flipped_image = cv2.flip(src=img, flipCode=1)
test_img = cv2.resize(img,(150,150))


print img.shape
t = img.shape
crp = img[0:t[0]*4/5,0:t[1]*4/5]
crp2 = img[t[0]*1/5:t[0],t[1]*1/5:t[1]]

crp3 = img[0:t[0],0:t[1]*4/5]
crp4 = img[0:t[0]*4/5,0:t[1]]
crp5 = img[t[0]*1/5:t[0]*4/5,t[1]*1/5:t[1]*4/5]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow('crp',crp)
cv2.imshow('crp2',crp2)
cv2.imshow('crp3',crp3)
cv2.imshow('crp4',crp4)
cv2.imshow('crp5',crp5)

cv2.imshow('crp5',gray)

'''cv2.imshow('testimg',test_img)
cv2.imshow('img',flipped_image)
cv2.imshow('img2',img)'''
cv2.waitKey(0)
cv2.destroyAllWindows()
