from drawme import draw_lmks_on_original
from utils import *
from drawme import  fit_to_screen
import math
from scipy import linspace
import random
import argparse
from scipy.ndimage import morphology
import matplotlib.pyplot as plt
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

    npcs, _ = index_of_true(variance_explained > 0.99)
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
    #print('lmks: {}'.format(lmks))
    #print('tpoint: {}'.format(tpoint))
    p_prev = lmks[(pidx - 1) % 40 , :]
    p_next = lmks[(pidx + 1) % 40, :]
    n1 = np.array([p_prev[1] - tpoint[1], tpoint[0] - p_prev[0]])
    n2 = np.array([tpoint[1] - p_next[1], p_next[0] - tpoint[0]])
    n = (n1 + n2) / 2
    return n / np.linalg.norm(n)


def create_grayscaleimage(img):
    """CLAHE filter first, then blur and then sobel to get grayscale gradient image
    https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    :param img:
    :return:
    """
    # i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahef = cv2.createCLAHE(2.0, (32,32))
    # i = clahef.apply(i)
    i = cv2.GaussianBlur(img, (3, 3), 0)
    sobelx = cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def enhance_image(img):
    """
    https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html

    :param img:
    :return:
    """
    i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    i = cv2.medianBlur(i, 5)
    i = cv2.bilateralFilter(i, 9, 75, 75)
    i_top_hat = morphology.white_tophat(i, size=400)
    i_bot_hat = morphology.black_tophat(img, size=80)
    i = cv2.add(i, i_top_hat)
    i = cv2.subtract(i, i_bot_hat)
    clahef = cv2.createCLAHE(2.0, (32,32))
    i = clahef.apply(i)
    return i


def create_pyramids_and_stuff(images, levels, lmks, k):
    """Create a gaussian pyramid of a given image.

    Args:
        image: The source image.
        levels (int): The number of pyramid levels.

    Returns:
        A list of images, the original image as first and the most scaled down one
        as last.

    """
    pyramids = []
    for image in images:
        layers = []
        layers.append(image)
        tmp_image = image
        for _ in range(0, levels):
            tmp_image = cv2.pyrDown(tmp_image)
            layers.append(tmp_image)
        pyramids.append(layers)
    lmks_per_pyramid = [[dlmk.dot(1/2**i) for i in range(levels+1)] for dlmk in lmks]


    mean_covariances = []

    for did1 in range(levels + 1):
        enhaced_images = [enhance_image(img) for img in list(zip(*pyramids))[did1]]
        gimages = [create_grayscaleimage(img) for img in enhaced_images]
        tlmks = list(zip(*lmks_per_pyramid))[did1]
        dmean_covariances = []

        #print('tlmks: {}'.format(tlmks.shape))
        for did2 in range(40):

            dmean_covariances.append(build_greyscale_model(enhaced_images, gimages, tlmks, did2, k))

        mean_covariances.append(dmean_covariances)


    return pyramids, lmks_per_pyramid, mean_covariances




def get_points_along_normal(img, gradient_img, point, normal, k):
    #print('point: {}'.format(point))
    a = point
    b = point + normal * k
    # print('a: {}'.format(a))
    # print('b: {}'.format(b))
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
    mean = (np.mean(tmatrix, axis=0))
    covariance = (np.cov(tmatrix, rowvar=False))

    return mean, covariance

def fit_quality(mean, covariance, samples):
    return (samples - mean).T.dot(covariance).dot(samples - mean)


def find_fits(lmk, img, gimg, mean_covariances, m, k):
    bests = []
    fits = []
    points = []
    number_in_50 = 0
    for did in range(len(lmk)):
        tnormal = normal_to_one_point(lmk, did)
        tpoint = lmk[did, :]
        tpoints , tsamples = sample_along_normal(img, gimg, tpoint, tnormal, m)
        # print('tsamples: {}'.format(len(tsamples)))
        # print('m: {}'.format(m))

        # print('tsamples: {}'.format(tsamples))
        # print('tpoints: {}'.format(tpoints))
        #sys.exit(0)

        # max_value = max(tsamples)
        # max_index = tsamples.index(max_value)
        # bests.append(max_index)
        # points.append(tpoints)

        dmin, best = np.inf, None
        dists = []
        for i in range(k, k + 2 * (m - k) + 1):
            subprofile = tsamples[i - k:i + k + 1]

            tmean, tcov = mean_covariances[did]
            dist = fit_quality(tmean, tcov, subprofile)
            dists.append(dist)
            if dist < dmin:
                dmin = dist
                best = i
        bests.append(best)
        points.append(tpoints)
        if 3/4 * m < best < 5/4 * m:
            number_in_50 += 1
    # print('bests: {}'.format(bests))
    # print('len tsamples: {}'.format(len(tsamples)))
    # print('len bests: {}'.format(len(bests)))
    # print('m: {}'.format(m))
    # print('number_in_50: {}'.format(number_in_50))
    ratio_in_50 = number_in_50/len(bests)
    for did in range(len(points)):
        fits.append([int(x) for x in points[did][bests[did], :]])
    return fits, ratio_in_50





def fit_one(estimate, image_to_run_on, gimage_to_run_on, m, pc_modes, mean_covariances, k):

    total_s = 1
    total_theta = 0
    X = estimate
    for _ in range(50):
        Y, ratio_in_50 = find_fits(X, image_to_run_on, gimage_to_run_on, mean_covariances, m, k)
        print('current iteration/ratio: {}/{}'.format(_, ratio_in_50))
        if ratio_in_50 >= 0.65:
            break
        Y = np.asarray(Y)
        #draw_lmks_on_original([X, Y], image_to_run_on, [[0,0,255], [0,255,0]])
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
        #print(theta)
        X = unflatten_special(flatten_special(X) + np.dot(pc_modes, b))
        X = transform(X, t, s, theta)

        draw_lmks_on_original([X], image_to_run_on, [[255,255,255]])
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
        y = inverse_transform(Y, t, s, theta)
        y1 = unflatten_special(flatten_special(y)/np.dot(flatten_special(y), flatten_special(X).T))
        b_prev = b
        b = np.dot(pc_modes.T, (flatten_special(y1) - flatten_special(X)))

    return b, t, s, theta


def stupid_init(lmk):
    #noise = [[random.randint(-10,+10), random.randint(-10,+10)] for _ in range(40)]
    noise = [[-10, +10] for _ in range(40)]
    test_estimate = lmk + np.array(noise)
    return test_estimate
# def main(tooth_number, rg_to_predict_on, lmk_location='../Landmarks/original/', rg_location='../Radiographs/'):
#     m = 30
#     k = 10
#     dataset = read_dataset(tooth_number, lmk_location, rg_location)
#     images = [x['image'] for x in dataset if x['rg'] != rg_to_predict_on]
#
#     enhanced_images = [enhance_image(img) for img in images]
#     gimages = [create_grayscaleimage(img) for img in enhanced_images]
#
#     for item in dataset:
#         if item['rg'] == rg_to_predict_on:
#             test_lmk = item['lmk']
#             test_img = item['image']
#             original_test_image = test_img.copy()
#             test_image = enhance_image(test_img)
#             test_gimg = create_grayscaleimage(test_img)
#             break
#
#
#     lmks = [x['lmk'] for x in dataset if x['rg'] != rg_to_predict_on]
#     mean_covariances = []
#     for i in range(40):
#         mean_covariances.append(build_greyscale_model(images, gimages, lmks, i, k))
#
#     mean, lmks_gpa = gpa(lmks)
#     pc_modes = get_pca(mean, lmks_gpa)
#     test_estimate = stupid_init(test_lmk)
#
#     Z = fit_one(test_estimate, test_image, test_gimg, m, pc_modes, mean_covariances, k)
#     draw_lmks_on_original([test_estimate, Z], original_test_image, [[0,0,255], [0,255,0]])

def main(tooth_number, rg_to_predict_on, lmk_location='../Landmarks/original/', rg_location='../Radiographs/', number_pyramids=2):
    m = 15
    k = 10
    dataset = read_dataset(tooth_number, lmk_location, rg_location)
    images = [x['image'] for x in dataset if x['rg'] != rg_to_predict_on]
    for item in dataset:
        if item['rg'] == rg_to_predict_on:
            test_lmk = item['lmk']
            test_img = item['image']
            original_test_image = test_img.copy()
            break
    lmks = [x['lmk'] for x in dataset if x['rg'] != rg_to_predict_on]
    mean, lmks_gpa = gpa(lmks)
    pc_modes = get_pca(mean, lmks_gpa)
    test_estimate = stupid_init(test_lmk)
    a, b, mean_covariances = create_pyramids_and_stuff(images, number_pyramids, lmks, k)

    Z = fit_pyramid(test_estimate, test_img, m, mean_covariances, pc_modes, k, number_pyramids=number_pyramids)
    draw_lmks_on_original([test_estimate, Z], original_test_image, [[0, 0, 255], [0, 255, 0]])



def fit_pyramid(estimate, image, m, mean_covariances, pc_modes, k, number_pyramids=3):
    pyramid = create_pyramid(image, number_pyramids)
    #print('pyramid: {}'.format(pyramid[1].shape))
    # print('estimate: {}'.format(estimate))
    lmk_pyramid = estimate.dot(1/2 ** (number_pyramids + 1))
    # print('lmk_pyramid: {}'.format(lmk_pyramid))
    print('len mean_covariances: {}'.format(len(mean_covariances)))
    print('len pyramid: {}'.format(len(pyramid)))
    for img, mean_cov in zip(reversed(pyramid), reversed(mean_covariances)):
        print('running one fit')
        # plt.imshow(img)
        # plt.show()
        # sys.exit(0)
        gimg = enhance_image(img)
        gimg = create_grayscaleimage(gimg)
        # print(lmk_pyramid)
        # draw_lmks_on_original([lmk_pyramid], img, [[255,255,255]])
        # sys.exit(0)
        lmk_pyramid = lmk_pyramid.dot(2)
        # print(lmk_pyramid)
        # sys.exit(0)
        lmk_pyramid = fit_one(lmk_pyramid, img, gimg, m, pc_modes, mean_cov, k)
    return lmk_pyramid


def create_pyramid(image, levels):
    output = []
    output.append(image)
    tmp = image
    for _ in range(0, levels):
        tmp = cv2.pyrDown(tmp)
        output.append(tmp)
    return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tooth', help='tooth number to predict', required=True, type=int)
    parser.add_argument('-r', '--rg', help='Radigraph to predict on', required=True, type=int)
    parser.add_argument('-g', '--rglocation', help='location of radigraphs', default='', type=str)
    parser.add_argument('-l', '--lmlocation', help='location of landmarks', default='', type=str)
    args = parser.parse_args()

    if len(args.rglocation) > 0 and len(args.lmlocation)>0:
        main(args.tooth, args.rg, lmk_location=args.lmlocation, rg_location=args.rglocation)
    elif len(args.rglocation) > 0:
        main(args.tooth, args.rg, rg_location=args.rglocation)
    elif len(args.lmlocation) > 0:
        main(args.tooth, args.rg, lmk_location=args.lmlocation)
    else:
        main(args.tooth, args.rg)

