
################# Comment this part if you don't have GPU ##########
import GPUtil
deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5, excludeID=[], excludeUUID=[])
if len(deviceIDs) == 1:

    devicetouse = deviceIDs[0]
    print('will be using GPU: {}'.format(devicetouse))
else:
    raise Exception('no GPU available')
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(devicetouse)
######################################################################
import os
import numpy as np
import sys
import shutil
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Lambda, Dropout, merge
from keras.layers.merge import concatenate
from keras.optimizers import Adamax, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import array_to_img

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

def to_rgb1(im):
    # I think this will be slow
    w, h, _ = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im[:,:,0]
    ret[:, :, 1] = im[:,:,0]
    ret[:, :, 2] = im[:,:,0]
    return ret.astype(np.uint8)

def get_big_unet(img_rows,img_cols, ks=3, mp=2):
    inputs = Input((img_rows, img_cols,3))
    inputs_norm = Lambda(lambda x: x/127.5 - 1.)
    norm_inp = inputs_norm(inputs)
    conv1 = Conv2D(8, (ks, ks), activation='relu', padding='same')(norm_inp)
    conv1 = Conv2D(8, (ks, ks), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(mp, mp))(conv1)

    conv2 = Conv2D(16, (ks, ks), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (ks, ks), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(mp, mp))(conv2)

    conv3 = Conv2D(32, (ks, ks), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (ks, ks), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(mp, mp))(conv3)

    conv4 = Conv2D(64, (ks, ks), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(64, (ks, ks), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(mp, mp))(conv4)

    conv5 = Conv2D(128, (ks, ks), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(128, (ks, ks), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(mp, mp))(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (ks, ks), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (ks, ks), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(mp, mp))(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (ks, ks), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (ks, ks), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(mp, mp))(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (ks, ks), activation='relu', padding='same')(up8)
    conv8 = Conv2D(16, (ks, ks), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(mp, mp))(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (ks, ks), activation='relu', padding='same')(up9)
    conv9 = Conv2D(8, (ks, ks), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)


    return model


IMG = sys.argv[1]
train_X = np.load('./datasets/train_X_' + IMG + '.npy')
train_Y = np.load('./datasets/train_Y_' + IMG + '.npy')
valid_X = np.load('./datasets/valid_X_' + IMG + '.npy')
valid_Y = np.load('./datasets/valid_Y_' + IMG + '.npy')

a = get_big_unet(1024,512, ks=7,mp=2)
a.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.00005)

a.compile(optimizer=RMSprop(lr=0.0001),
             loss=IOU_calc_loss, metrics=[IOU_calc, Intersec_round, Intersec])
a.fit(x=train_X, y=train_Y, validation_data=(valid_X, valid_Y), batch_size=16, epochs=700, callbacks=[TensorBoard(log_dir='./tb',write_graph=True, write_images=True), ModelCheckpoint(os.path.join("./models/", 'model_{epoch:02d}_{loss:.4f}_'+IMG+'.h5p'),save_best_only=False), reduce_lr])


a.save('segmentation_model'+IMG+'.h5')
