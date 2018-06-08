"""
This code creates data for training U-net. It's not optimized and takes a lot of time to run. Models are test images are pre-created and provided.
Variable RGPH needs to be changed every run to specify which RG we should skip, when creating data to have leave-one-out behaviour.
It also creates a lot of temporary images that are not removed.

"""


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
import numpy as np
from tqdm import tqdm
import random
import shutil
from utils import read_dataset
RGPH = '01.tif'

### This piece creates files in ./original/ - basically copy of radiographs and with their masks.
### it takes a while to run it, so preprocessed files are supplied together with project

lmk_location='./Landmarks/original/'
rg_location='./Radiographs/'
X = './original/X'
Y = './original/Y'


for rg in [1, 2, 3, 4, 5, 6, 7 ,8 ,9 , 10, 11 ,12 ,13, 14]:
    alllmks = {}
    for ttt in [1,2,3,4,5,6,7,8]:


        dataset = read_dataset(ttt, lmk_location, rg_location)
        for item in dataset:
            if item['rg'] == rg:
                if ttt in alllmks:
                    pass
                else:

                    alllmks[ttt] = item['lmk']
                test_img = item['image']
                break

    name = str(rg)

    if len(name) == 1:
        name = '0' + name + '.tif'

    else:
        name += '.tif'

    #testgray = cv2.cvtColor(test_img ,cv2.COLOR_RGB2GRAY)
    mask_image = np.zeros_like(test_img)
    for ttt in alllmks:
        dlmk = alllmks[ttt]
        dcnt = np.expand_dims(dlmk, axis=1).astype(int)
        cv2.drawContours(mask_image, [dcnt], 0, (255,255,255), cv2.FILLED)
    mask_image = cv2.cvtColor(mask_image ,cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite(os.path.join(Y, name), im_bw)
    cv2.imwrite(os.path.join(X, name), test_img)



############# CREATE CUT version of files later to be used as training images for U-NET
import cv2
DIR = './original/X/'
MSKS = './original/Y/'
files = [os.path.join(DIR, dfile) for dfile in os.listdir(DIR) if dfile.endswith('.tif')]
masks = [os.path.join(MSKS, dfile) for dfile in os.listdir(MSKS) if dfile.endswith('.tif')]
for dfile in files:
    dimg = cv2.imread(dfile)
    dimg = dimg[200:dimg.shape[0]-200, dimg.shape[1]//2 - 240:dimg.shape[1]//2 + 240]
    sfile = os.path.join('./cutme/X', os.path.basename(dfile))
    cv2.imwrite(sfile, dimg)
for dfile in masks:
    dimg = cv2.imread(dfile, 0)
    dimg = dimg[200:dimg.shape[0]-200:,dimg.shape[1]//2 - 240:dimg.shape[1]//2 + 240]
    sfile = os.path.join('./cutme/Y', os.path.basename(dfile))
    cv2.imwrite(sfile, dimg)



DIR = './cutme/X'
MSKS = './cutme/Y'
filesimg = sorted([os.path.join(DIR, dfile) for dfile in os.listdir(DIR) if dfile.endswith('.tif') and dfile != RGPH ])
filesmsk = sorted([os.path.join(MSKS, dfile) for dfile in os.listdir(MSKS) if dfile.endswith('.tif')  and dfile != RGPH])
resize_size = (562, 1124)
cut_y = 50
cut_x = 25
expected_y = 1024
expected_x = 512


img_datagen = ImageDataGenerator( rotation_range=0.1,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=0.05,
                                   fill_mode='constant',
                                )
msk_datagen = ImageDataGenerator( rotation_range=0.1,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=0.05,
                                   fill_mode='constant',
                                )

images = []
for dfile in filesimg:
    img = cv2.imread(dfile)
    img = cv2.resize(img, resize_size)
    images.append(img)
masks = []
for dfile in filesmsk:
    msk = cv2.imread(dfile, 0)
    msk = cv2.resize(msk, resize_size)
    msk = np.expand_dims(msk, 2)
    masks.append(msk)

print(len(masks))
print(len(images))

nimg = np.asarray(images)
print(nimg.shape)
nmsk = np.asarray(masks)
print(nmsk.shape)
print(nimg.shape)
cntr = 0

MCNT = 20
try:
    shutil.rmtree('./dataset_img/X/')
    shutil.rmtree('./dataset_img/Y/')
except FileNotFoundError:
    pass
os.mkdir('./dataset_img/X/')
os.mkdir('./dataset_img/Y/')
for did in tqdm(range(nimg.shape[0])):
    imgc = 0
    mskc = 0
    dimg = np.expand_dims(nimg[did], axis=0)
    dmsk = np.expand_dims(nmsk[did], axis=0)

    seed = random.randint(1, 100)
    print('seed: {}'.format(seed))
    if os.path.isdir('./dataset_img/X/' + str(did)):
        shutil.rmtree('./dataset_img/X/' + str(did))
    if os.path.isdir('./dataset_img/Y/' + str(did)):
        shutil.rmtree('./dataset_img/Y/' + str(did))
    os.mkdir('./dataset_img/X/' + str(did))
    os.mkdir('./dataset_img/Y/' + str(did))
    for i in img_datagen.flow(dimg, batch_size=1, save_to_dir='./dataset_img/X/' + str(did), save_prefix=str(did) + '_' , save_format='tif', seed=seed):
        imgc += 1
        if imgc >= MCNT:
            break
    for i in msk_datagen.flow(dmsk, batch_size=1, save_to_dir='./dataset_img/Y/' + str(did), save_prefix=str(did) + '_', save_format='tif', seed=seed):
        mskc += 1
        if mskc >= MCNT:
            break


TIMG = './dataset_img/X/'
TMSK = './dataset_img/Y/'
TIMG_INT_DIR = [os.path.join(TIMG, intdir) for intdir in os.listdir(TIMG) if os.path.isdir(os.path.join(TIMG, intdir))]
TMSK_INT_DIR = [os.path.join(TMSK, intdir) for intdir in os.listdir(TMSK) if os.path.isdir(os.path.join(TMSK, intdir))]
timg = []
tmsk = []
for ddir in TIMG_INT_DIR:
    timg += [os.path.join(ddir, f) for f in os.listdir(ddir)]
for ddir in TMSK_INT_DIR:
    tmsk += [os.path.join(ddir, f) for f in os.listdir(ddir)]

for dfile in tqdm(tmsk):
    dmsk = cv2.imread(dfile, 0)
    #print(dmsk.shape)
    dmsk = dmsk[cut_y:cut_y+expected_y, cut_x:expected_x + cut_x]
    (thresh, im_bw) = cv2.threshold(dmsk, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(dfile, im_bw)

for dfile in tqdm(timg):
    dimg = cv2.imread(dfile, 0)
    dimg = dimg[cut_y:cut_y+expected_y, cut_x:expected_x + cut_x]
    cv2.imwrite(dfile, dimg)

TIMG = './dataset_img/X/'
TMSK = './dataset_img/Y/'
TIMG_INT_DIR = [os.path.join(TIMG, intdir) for intdir in os.listdir(TIMG) if os.path.isdir(os.path.join(TIMG, intdir))]
TMSK_INT_DIR = [os.path.join(TMSK, intdir) for intdir in os.listdir(TMSK) if os.path.isdir(os.path.join(TMSK, intdir))]
timg = []
tmsk = []
for ddir in TIMG_INT_DIR:
    timg += [os.path.join(ddir, f) for f in os.listdir(ddir)]
for ddir in TMSK_INT_DIR:
    tmsk += [os.path.join(ddir, f) for f in os.listdir(ddir)]
timg = sorted(timg)
tmsk = sorted(tmsk)

dataset_images = []
dataset_masks = []
for dfile in tqdm(timg):
    dimg = cv2.imread(dfile)
    dataset_images.append(dimg)
for dfile in tqdm(tmsk):

    dmsk = cv2.imread(dfile, 0)
#    print(dmsk.shape)
    dmsk[dmsk == 255] = 1
    dataset_masks.append(dmsk)

idexes = [x for x in range(len(dataset_images))]
random.shuffle(idexes)

dataset_images_shuffled = []
dataset_masks_shuffled = []

for did in idexes:
    dataset_images_shuffled.append(dataset_images[did])
    dataset_masks_shuffled.append(dataset_masks[did])

train_images = dataset_images_shuffled[:int(0.8 * len(dataset_images_shuffled))]
train_masks = dataset_masks_shuffled[:int(0.8 * len(dataset_images_shuffled))]
valid_images = dataset_images_shuffled[int(0.8 * len(dataset_images_shuffled)):]
valid_masks = dataset_masks_shuffled[int(0.8 * len(dataset_images_shuffled)):]
train_X = np.asarray(train_images)
train_Y = np.expand_dims(np.asarray(train_masks),  axis=3)
valid_X = np.asarray(valid_images)
valid_Y = np.expand_dims(np.asarray(valid_masks), axis=3)

print(train_X.shape)
print(train_Y.shape)
print(valid_X.shape)
print(valid_Y.shape)
np.save('./datasets/train_X_' + RGPH.split('.')[0] +'.npy', train_X)
np.save('./datasets/train_Y_' + RGPH.split('.')[0] +'.npy', train_Y)
np.save('./datasets/valid_X_' + RGPH.split('.')[0] +'.npy', valid_X)
np.save('./datasets/valid_Y_' + RGPH.split('.')[0] +'.npy', valid_Y)

TEST_DIR = './testimages/'
DIR = './cutme/X/'
testfilesimg = sorted([os.path.join(DIR, dfile) for dfile in os.listdir(DIR) if dfile.endswith('.tif')])
for dfile in testfilesimg:
    img = cv2.imread(dfile)
    targetfile = os.path.join(TEST_DIR, os.path.basename(dfile))
    #    print(img.shape)
    # cv2.imshow('original', img)
    # cv2.waitKey(0)
    img = cv2.resize(img, resize_size)
    img = img[cut_y:cut_y + expected_y, cut_x:expected_x + cut_x]
    print(img.shape)
    cv2.imwrite(targetfile, img)
