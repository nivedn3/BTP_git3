import cv2
from glob import glob
import os
import numpy as np

ite = 0
data_raw = glob(os.path.join('/home//psycholearner/projects/DCGAN-tensorflow/data/celebA','*.jpg'))
data_train = sorted(data_raw)

a = [26, 31, 59, 144, 161, 180, 199, 213, 221, 235, 265, 310, 335, 346, 380, 383, 395, 439, 465, 496, 499, 502, 507, 521, 544, 546, 556, 557, 560, 585, 590, 599, 638, 668, 711, 756, 764, 778, 782, 806, 844, 866, 880, 894, 914, 928, 930, 957, 959, 961, 965, 968, 977, 1054, 1061, 1086, 1099, 1170, 1216, 1218, 1254, 1276, 1282, 1325, 1338, 1340, 1373, 1390, 1435, 1465, 1491, 1556, 1565, 1581, 1617, 1719, 1732, 1746, 1751, 1772, 1787, 1869, 1894, 1906, 1909, 2028, 2053, 2059, 2118, 2120, 2129, 2155, 2166, 2172, 2180, 2192, 2204, 2247, 2250, 2264, 2337, 2373, 2422, 2446, 2489, 2505, 2542]

for i in a:
	img = cv2.imread(data_train[200000+i])
	cv2.imshow('crp5',img)
	cv2.waitKey(0)

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


#cv2.imshow('crp',crp)
#cv2.imshow('crp2',crp2)
#cv2.imshow('crp3',crp3)
#cv2.imshow('crp4',crp4)
#cv2.imshow('crp5',crp5)

cv2.imshow('crp5',img)

'''cv2.imshow('testimg',test_img)
cv2.imshow('img',flipped_image)
cv2.imshow('img2',img)'''
cv2.waitKey(0)
cv2.destroyAllWindows()
