import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm
import pickle


def read_landmarks(tooth, rg_to_skip=None, lmks_dir='../Landmarks/original'):
    remove_filename = []
    if rg_to_skip is not None:
        for ditem in rg_to_skip:
            remove_filename.append('landmarks' + str(ditem) + '-')
    non_filtered_files = [dfile for dfile in glob(lmks_dir + '/landmarks*-{}.txt'.format(tooth))]
    filtered_files = []
    drop_idx = set()
    for ditem1 in remove_filename:
        for ditem_idx in range(len(non_filtered_files)):
            if os.path.basename(non_filtered_files[ditem_idx]).startswith(ditem1):
                drop_idx.add(ditem_idx)
    for ditem_idx in range(len(non_filtered_files)):
        if ditem_idx not in drop_idx:
            filtered_files.append(non_filtered_files[ditem_idx])

    landmarks_list = []
    filtered_files = ['../Landmarks/original/landmarks1-1.txt']
    for dfile in filtered_files:
        tmp_list_all = []
        with open(dfile) as fr:
            all_lines = fr.readlines()
            for ditem_idx in range(0, len(all_lines), 2):
                tmp_list_all.append([float(all_lines[ditem_idx].strip()), float(all_lines[ditem_idx + 1].strip())])
        landmarks_list.append(np.asarray(tmp_list_all))

    return landmarks_list


def read_dataset(tooth, lmk_location='../Landmarks/original/', rg_location='../Radiographs/'):
    print('reading dataset,')
    rgfs = glob(rg_location + '*.tif')
    all_avialble_rgfs = [int(os.path.basename(x).split('.', 1)[0]) for x in rgfs]
    results = []
    with tqdm(total=len(all_avialble_rgfs)) as pbar:
        for ditem2 in all_avialble_rgfs:
            pbar.set_description('processing RG {}'.format(ditem2))
            rg_tfile = rg_location + str(ditem2).zfill(2) + '.tif'
            lmk_tfile = lmk_location + 'landmarks{}-{}.txt'.format(ditem2, tooth)
            tlmks = []
            with open(lmk_tfile) as fr:
                all_lines = fr.readlines()
                for ditem_idx in range(0, len(all_lines), 2):
                    tlmks.append([float(all_lines[ditem_idx].strip()), float(all_lines[ditem_idx + 1].strip())])
            tlmks = np.asarray(tlmks)
            tmpdict = {'tooth': tooth,
                       'rg': ditem2,
                       'image': cv2.imread(rg_tfile),
                       'lmk': tlmks}
            results.append(tmpdict)
            pbar.update()
    results = sorted(results, key=lambda x: x['rg'])
    return results






def flatten_special(lmk):
    """
    helper to convert numpy array of shape (40,2)/[[x1,y1],...[xn-yn] into (80,)/[x1,..xn, y1,...yn]
    """
    return np.hstack((lmk[:,0], lmk[:,1]))

def unflatten_special(lmk):
    """
    reverse for flatten_special

    """
    return np.array((lmk[:len(lmk) // 2], lmk[len(lmk) // 2:])).T


def translate_to_origin(lmk):
    """
    Protocol 4, p1
    """
    #print(lmk.shape)
    center = np.mean(lmk, axis=0)
    return lmk - center


def rotate_lmk(lmk, angle):

    rmatrix = np.array([[np.cos(angle), np.sin(angle)],
                       [-np.sin(angle), np.cos(angle)]])

    tmp_matrix = np.zeros_like(lmk)
    center = np.mean(lmk, axis=0)
    centered_lmk = lmk - center
    for ditem_idx in range(len(centered_lmk)):
        try:
            tmp_matrix[ditem_idx, :] = centered_lmk[ditem_idx, :].dot(rmatrix)
        except:
            print(ditem_idx)
            raise
    tmp_matrix += center

    return tmp_matrix


def scale_by_param(lmk, param):
    center = np.mean(lmk, axis=0)
    return (lmk - center).dot(param) + center


def scale_to_unit(lmk):
    """
    protocol 4, p2

    """
    center = np.mean(lmk, axis=0)
    scale_factor = np.sqrt(np.power(lmk - center, 2).sum())
    return lmk.dot(1. / scale_factor)


def get_center(lmk):
    return [lmk[:, 0].min() + (lmk[:, 0].max() - lmk[:, 0].min()) / 2,
            lmk[:, 1].min() + (lmk[:, 1].max() - lmk[:, 1].min()) / 2]


if __name__ == '__main__':
    print(read_dataset())