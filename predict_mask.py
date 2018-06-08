"""

to run :
python3 predict_mask.py ./testimages/01.tif ./models/segmentation_model01.h5



"""



import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5" #comment it out or adjust depending on GPUs you have
from keras.models import *
import numpy as np
import sys
import os
from keras import backend as K
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

smooth = 1.
def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def Intersec_round(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = (K.sum(y_true_f * y_pred_f) + smooth) / (K.sum(y_true_f) + smooth)
    return intersection

def Intersec(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = (K.sum(y_true_f * y_pred_f) + smooth) / (K.sum(y_true_f) + smooth)
    return intersection

def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)

tst_img = cv2.imread(sys.argv[1])
print(tst_img.shape)
#tst_img = np.expand_dims(tst_img, axis=2)
#print(tst_img.shape)
tst_img = np.asarray([tst_img])
print(tst_img.shape)

model = load_model(sys.argv[2], {'IOU_calc':IOU_calc, 'Intersec_round': Intersec_round, 'Intersec':Intersec,  'IOU_calc_loss': IOU_calc_loss})
vv = model.predict(tst_img)
print(vv.shape)
vv = np.squeeze(vv)
print(vv.shape)
#vv = np.around(vv)
#np.count_nonzero(vv)
vv[vv > 0.5] = 255

vv[vv<=0.5] = 0
#vv = np.expand_dims(vv, axis=2)
print(vv.shape)
print(np.count_nonzero(vv))
savename = os.path.join('./predicted/', 'predicted_mask'+ os.path.basename(sys.argv[1]) + '.tif')
print(savename)
#cv2.imwrite(savename, vv)
mpimg.imsave(savename, vv, cmap=plt.get_cmap('gray'))
#plt.imsave(fname=savename, arr=vv, cmap='Greys')
print('saved in ./predicted/predicted_mask{}.tif'.format(os.path.basename(sys.argv[1])))