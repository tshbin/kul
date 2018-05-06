from drawme import draw_lmks_on_original
from utils import *
import math
from scipy import linspace
import random
import argparse
######################## Global Procrustes analysis #######################
def gpa(lmks, margin_error = 1e-10):
    """
    protocol 4

    """
    #p1
    lmks = [translate_to_origin(dlmk) for dlmk in lmks]

    #p2
    mean_lmk = scale_to_unit(lmks[0])
    rolling_error = 1
    while rolling_error >= margin_error:
        #print(rolling_error)
        #p4
        lmks = [align_lmks(dlmk, mean_lmk) for dlmk in lmks]
        new_mean_lmk = calcuate_new_mean(lmks)
        new_mean_lmk = align_lmks(new_mean_lmk, mean_lmk)
        new_mean_lmk = scale_to_unit(new_mean_lmk)
        new_mean_lmk = translate_to_origin(new_mean_lmk)
        #print(new_mean_lmk.shape)
        #draw_gpa(new_mean_lmk, lmks)
        rolling_error = np.abs((flatten_special(mean_lmk) - flatten_special(new_mean_lmk))).mean()

        #print(rolling_error)
        mean_lmk = new_mean_lmk
        #break
    #draw_gpa(new_mean_lmk, lmks)
    return new_mean_lmk, lmks


def calcuate_new_mean(lmks):
    tmat = []
    for ditem in lmks:
        tmat.append(flatten_special(ditem))
    mat = np.array(tmat)
    return unflatten_special(np.mean(mat, axis=0))


def align_params(lmk1, lmk2):
    """
    Appendix D
    """
    lmk1 = flatten_special(lmk1)
    lmk2 = flatten_special(lmk2)
    l1 = len(lmk1)//2
    l2 = len(lmk2)//2
    lmk1_center = np.array([np.mean(lmk1[:l1]), np.mean(lmk1[l1:])])
    lmk2_center = np.array([np.mean(lmk2[:l2]), np.mean(lmk2[l2:])])
    lmk1 = [x - lmk1_center[0] for x in lmk1[:l1]] + [y - lmk1_center[1] for y in lmk1[l1:]]
    lmk2 = [x - lmk2_center[0] for x in lmk2[:l2]] + [y - lmk2_center[1] for y in lmk2[l2:]]

    norm_lmk1 = np.linalg.norm(lmk1) ** 2
    a = np.dot(lmk1, lmk2)/norm_lmk1
    b = (np.dot(lmk1[:l1], lmk2[l2:]) - np.dot(lmk1[l1:], lmk2[:l2])) / norm_lmk1
    s = np.sqrt(a ** 2 + b ** 2)
    theta = np.arctan(b/a)
    t = lmk2_center - lmk1_center

    return t, s, theta


def align_lmks(lmk1, lmk2):
    """
    Protocol 4, p4

    """
    t, s , theta = align_params(lmk1, lmk2)
    lmk1 = rotate_lmk(lmk1, theta)
    lmk1 = scale_by_param(lmk1, s)

    projection = np.dot(flatten_special(lmk1), flatten_special(lmk2))
    aligned = flatten_special(lmk1) * (1.0/projection)
    #convert back to (40,2)
    aligned = unflatten_special(aligned)
    return aligned
######################## END OF Global Procrustes analysis #######################

########################## Build adaptive state model ############################

def get_pca(mean_lmk, lmks):
    # covariance calculation
    tmp_mat = []
    for ditem in lmks:
        tmp_mat.append(flatten_special(ditem))

    lmks_as_matrix = np.array(tmp_mat)

    covmat = np.cov(lmks_as_matrix, rowvar=False)

    # PCA on shapes
    eigvals, eigvecs = np.linalg.eigh(covmat)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    variance_explained = np.cumsum(eigvals / np.sum(eigvals))

    # Build modes for up to 98% variance
    def index_of_true(arr):
        for index, item in enumerate(arr):
            if item:
                return index, item

    npcs, _ = index_of_true(variance_explained > 0.98)
    npcs += 1

    M = []
    # print('npcs: {}'.format(npcs))
    for i in range(0, npcs - 1):
        M.append(np.sqrt(eigvals[i]) * eigvecs[:, i])
    pc_modes = np.array(M).squeeze().T
    return pc_modes


def normal_to_one_point(lmks, pidx):
    # gradient is normal to the tangible line at that point
    # line eq = x(y2-y1) - y(x2-x1) + C = 0
    # gradient = (y2-y1, -x2 -x1)


    tpoint = lmks[pidx, :]
    p_prev = lmks[(pidx - 1) % 40 , :]
    p_next = lmks[(pidx + 1) % 40, :]
    n1 = np.array([p_prev[1] - tpoint[1], tpoint[0] - p_prev[0]])
    n2 = np.array([tpoint[1] - p_next[1], p_next[0] - tpoint[0]])
    n = (n1 + n2) / 2
    return n / np.linalg.norm(n)


def create_grayscaleimage(img):
    """CLAHE filter first, then blur and then sobel to get grayscale gradient image

    :param img:
    :return:
    """
    i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahef = cv2.createCLAHE(2.0, (32,32))
    i = clahef.apply(i)
    i = cv2.GaussianBlur(i, (3, 3), 0)
    sobelx = cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def get_points_along_normal(img, gradient_img, point, normal, k):
    a = point
    b = point + normal * k
    coordinates = (a[:, None] * linspace(1, 0, k + 1) +
                   b[:, None] * linspace(0, 1, k + 1))
    values = img[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
    grad_values = gradient_img[coordinates[1].astype(np.int), coordinates[0].astype(np.int)]
    return coordinates.T, values, grad_values


def sample_along_normal(img, gradient_img, point, normal, k):
    pos_points, pos_values, pos_grads = get_points_along_normal(img, gradient_img, point, -normal, k)
    neg_points, neg_values, neg_grads = get_points_along_normal(img, gradient_img, point, normal, k)


    neg_values = neg_values[::-1]  # reverse
    neg_grads = neg_grads[::-1]  # reverse
    neg_points = neg_points[::-1]  # reverse
    points = np.vstack((neg_points, pos_points[1:, :]))
    values = np.append(neg_values, pos_values[1:])
    grads = np.append(neg_grads, pos_grads[1:])
    div = max(sum([math.fabs(v) for v in values]), 1)
    samples = [float(g) / div for g in grads]
    return points, samples


def build_greyscale_model(imgs, gimgs, lmks, pointidx, k):
    samples = []
    for did in range(len(imgs)):
        tnormal = normal_to_one_point(lmks[did], pointidx)
        tpoint = lmks[did][pointidx, :]
        _, smpl = sample_along_normal(imgs[did], gimgs[did], tpoint, tnormal, k)
        samples.append(smpl)
    tmatrix = np.array(samples)
    mean = np.mean(tmatrix, axis=0)
    covariance = np.cov(tmatrix, rowvar=False)

    return mean, covariance

def fit_quality(mean, covariance, samples):
    return (samples - mean).T.dot(covariance).dot(samples - mean)

def find_fits(lmk, img, gimg, mean_covariances, m, k):
    bests = []
    fits = []
    points = []
    for did in range(len(lmk)):
        tnormal = normal_to_one_point(lmk, did)
        tpoint = lmk[did, :]
        tpoints , tsamples = sample_along_normal(img, gimg, tpoint, tnormal, m)
        dmin, best = np.inf, None
        dists = []
        for i in range(k, k + 2 * (m - k) + 1):
            subprofile = tsamples[i - k:i + k + 1]

            tmean, tcov = mean_covariances[i]
            dist = fit_quality(tmean, tcov, subprofile)
            dists.append(dist)
            if dist < dmin:
                dmin = dist
                best = i
        bests.append(best)
        points.append(tpoints)
    for did in range(len(bests)):
        fits.append([int(x) for x in points[did][bests[did], :]])
    return fits


def fit_one(estimate, image_to_run_on, gimage_to_run_on, m, pc_modes, mean_covariances, k):

    b = np.zeros(pc_modes.shape[1])
    total_s = 1
    total_theta = 0
    X = estimate
    for _ in range(30):
        Y = find_fits(X, image_to_run_on, gimage_to_run_on, mean_covariances, m, k)
        Y = np.asarray(Y)
        b, t, s, theta = update_parameters(X, Y, pc_modes)
        b = np.clip(b, -3, 3)
        s = np.clip(s, 0.95, 1.05)
        if total_s * s > 1.20 or total_s * s < 0.8:
            s = 1
        total_s *= s
        theta = np.clip(theta, -math.pi / 8, math.pi / 8)
        if total_theta + theta > math.pi / 4 or total_theta + theta < - math.pi / 4:
            theta = 0
        total_theta += theta
        X_prev = X

        X = unflatten_special(flatten_special(X) + np.dot(pc_modes, b))
        X = transform(X, t, s, theta)

        #draw_lmks_on_original([X_prev, X], image_to_run_on, title='MyTest')
    return X
def inverse_transform(X, t, s, theta):
    tt = X - t
    tt = scale_by_param(tt, 1/s)
    tt = rotate_lmk(tt, -theta)
    return tt

def transform(X, t, s, theta):
    tt = rotate_lmk(X, theta)
    tt = scale_by_param(tt, s)
    tt = tt + t
    return tt

def update_parameters(X, Y, pc_modes):
    b = np.zeros(pc_modes.shape[1])
    b_prev = np.ones(pc_modes.shape[1])
    i = 0
    while (np.mean(np.abs(b - b_prev)) >= 1e-14):
        x = unflatten_special(flatten_special(X) + np.dot(pc_modes, b))
        t, s, theta = align_params(x, Y)
        #print('Ybig: {}\n------------'.format(Y))
        # print('---------------------\nt: {}'.format(t))
        # print('s: {}'.format(t))
        # print('thera: {}'.format(theta))
        # print('Ybig: {}'.format(Y))
        y = inverse_transform(Y, t, s, theta)
        # print('ysmall: {}\n----------------------------'.format(y))
        # sys.exit(0)
        y1 = unflatten_special(flatten_special(y)/np.dot(flatten_special(y), flatten_special(X).T))
        b_prev = b
        b = np.dot(pc_modes.T, (flatten_special(y1) - flatten_special(X)))
        #print('current b: {}'.format(b))
    return b, t, s, theta


def stupid_init(lmk):
    noise = [[random.randint(-5,+5), random.randint(-5,+5)] for _ in range(40)]
    test_estimate = lmk + np.array(noise)
    return test_estimate
def main(tooth_number, rg_to_predict_on):
    dataset = read_dataset(tooth_number)
    images = [x['image'] for x in dataset if x['rg'] != rg_to_predict_on]
    gimages = [create_grayscaleimage(img) for img in images]

    for item in dataset:
        if item['rg'] == rg_to_predict_on:
            test_lmk = item['lmk']
            test_img = item['image']
            test_gimg = create_grayscaleimage(test_img)
            break


    lmks = [x['lmk'] for x in dataset if x['rg'] != rg_to_predict_on]
    mean_covariances = []
    for i in range(40):
        mean_covariances.append(build_greyscale_model(images, gimages, lmks, i, 10))

    mean, lmks_gpa = gpa(lmks)
    pc_modes = get_pca(mean, lmks_gpa)
    #noise = [[random.randint(-10,10), random.randint(-10,10)] for _ in range(40)]
    test_estimate = stupid_init(test_lmk)

    Z = fit_one(test_estimate, test_img, test_gimg, 15, pc_modes, mean_covariances, 10)
    draw_lmks_on_original([test_estimate, Z], test_img, [[0,0,255], [0,255,0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tooth', help='tooth number to predict', required=True, type=int)
    parser.add_argument('-r', '--rg', help='Radigraph to predict on', required=True, type=int)
    args = parser.parse_args()
    main(args.tooth, args.rg)
