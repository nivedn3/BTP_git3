import cv2
from glob import glob
import os

ite = 0
data_raw = glob(os.path.join('/home//psycholearner/projects/DCGAN-tensorflow/data/celebA','*.jpg'))
data_train = sorted(data_raw)[ite*10000:10000 + ite*10000]

img = cv2.imread(data_train[108])


flipped_image = cv2.flip(src=img, flipCode=1)


cv2.imshow('img',flipped_image)
cv2.imshow('img2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()