from drawme import draw_lmks_on_original
from utils import *
import math
from scipy import linspace
import argparse
from scipy.ndimage import morphology
import csv
from autoinit import scale_mask_back

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
    """basically from ASM we need only 2 things, mean and PC vectors.


    :param mean_lmk:
    :param lmks:
    :return:
    """
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
    best_X = None
    best_ratio = -np.inf
    for _ in range(50):
        Y, ratio_in_50 = find_fits(X, image_to_run_on, gimage_to_run_on, mean_covariances, m, k)
        print('current iteration/ratio: {}/{}'.format(_, ratio_in_50))
        Y = np.asarray(Y)
        #draw_lmks_on_original([X, Y], image_to_run_on, [[0,0,255], [0,255,0]])
        b, t, s, theta = update_parameters(X, Y, pc_modes)
        b = np.clip(b, -3, 3)
        s = np.clip(s, 0.95, 1.05)
        if total_s * s > 1.20 or total_s * s < 0.8:
            s = 1
        total_s *= s
        theta = np.clip(theta, -math.pi / 30, math.pi / 30)
        if total_theta + theta > math.pi / 8 or total_theta + theta < - math.pi / 8:
            theta = 0
        total_theta += theta
        #print(theta)
        X = unflatten_special(flatten_special(X) + np.dot(pc_modes, b))
        X = transform(X, t, s, theta)
        if ratio_in_50 >= 0.6:
            return X
        if ratio_in_50 > best_ratio:
            best_ratio = ratio_in_50
            best_X = X
        #draw_lmks_on_original([X], image_to_run_on, [[255,255,255]])
    return best_X

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


# def stupid_init(lmk):
#     #noise = [[random.randint(-10,+10), random.randint(-10,+10)] for _ in range(40)]
#     noise = [[-10, +10] for _ in range(40)]
#     test_estimate = lmk + np.array(noise)
#     return test_estimate

def stupid_init(toothN, rg):
    #noise = [[random.randint(-10,+10), random.randint(-10,+10)] for _ in range(40)]
    #noise = [[-10, +10] for _ in range(40)]
    #test_estimate = lmk + np.array(noise)
    rgstr = str(rg)
    if len(rgstr) == 2:
        pass
    else:
        rgstr = '0' + rgstr
    mskfile = './predicted/predicted_mask{}.tif.tif'.format(rgstr)
    origorig = './Radiographs/{}.tif'.format(rgstr)
    du, dl = scale_mask_back(mskfile, origorig, rg)
    inilmks = du + dl
    # with open('initial_lmks_{}.cpkl'.format(rgstr), 'rb') as fr:
    #     inilmks = pickle.load(fr)
    test_estimate = inilmks[toothN - 1]
    return test_estimate[0][0], test_estimate[0][1], test_estimate[1], test_estimate[3], test_estimate[2], test_estimate[-1]


def main(tooth_number, rg_to_predict_on, lmk_location='./Landmarks/original/', rg_location='./Radiographs/', number_pyramids=1):
    # m = 40
    # k = 30
    m = 30
    k = 20
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
    cx, cy, h, angle , dcnt, w = stupid_init(tooth_number, rg_to_predict_on)
    sparam = h / (mean[:, 1].max() - mean[:, 1].min())
    mean_scaled = scale_by_param(mean, sparam)
    mean_moved = move_by_vec(mean_scaled, np.array([cx, cy]))

    if angle is not None:
        mean_moved_cnt = np.expand_dims(mean_moved, axis=1).astype(int)
        rect = cv2.minAreaRect(mean_moved_cnt)
        mean_angle = rect[2]
        if rect[1][0] < rect[1][1]:
            mean_angle = mean_angle - 90

        rotate_angl = angle - mean_angle

        rotate_radians = rotate_angl * (math.pi / 180)
        mean_moved_rotated = rotate_lmk(mean_moved, rotate_radians)
    else:
        mean_moved_rotated = mean_moved
    a, b, mean_covariances = create_pyramids_and_stuff(images, number_pyramids, lmks, k)
    Z = fit_pyramid(mean_moved_rotated, test_img, m, mean_covariances, pc_modes, k, number_pyramids=number_pyramids)
    cleanimage = np.zeros_like(original_test_image)
    shape1 = cleanimage.copy()
    shape2 = cleanimage.copy()
    rrrZ = np.expand_dims(Z, axis=1).astype(int)
    rrr_test_lmk = np.expand_dims(test_lmk, axis=1).astype(int)
    cv2.drawContours(shape1, [rrrZ], 0, (100 ,0,0), cv2.FILLED)
    cv2.drawContours(shape2, [rrr_test_lmk], 0, (200, 0, 0), cv2.FILLED)
    shape1 = cv2.cvtColor(shape1 ,cv2.COLOR_RGB2GRAY)
    shape2 = cv2.cvtColor(shape2 ,cv2.COLOR_RGB2GRAY)
    unique, counts = np.unique(shape1, return_counts=True)
    prediction = dict(zip(unique, counts))[30]

    unique, counts = np.unique(shape2, return_counts=True)
    truth = dict(zip(unique, counts))[60]

    together = shape1 + shape2
    unique, counts = np.unique(together, return_counts=True)
    intersection = dict(zip(unique, counts))[90]
    precision = intersection/prediction
    recall = intersection/truth
    F1 = 2 * (precision * recall)/(precision + recall)
    return test_lmk, Z, F1, original_test_image


def fit_pyramid(estimate, image, m, mean_covariances, pc_modes, k, number_pyramids=1):
    pyramid = create_pyramid(image, number_pyramids)
    lmk_pyramid = estimate.dot(1/2 ** (number_pyramids + 1))
    for img, mean_cov in zip(reversed(pyramid), reversed(mean_covariances)):
        print('running one fit')
        gimg = enhance_image(img)
        gimg = create_grayscaleimage(gimg)
        lmk_pyramid = lmk_pyramid.dot(2)
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
    parser.add_argument('-t', '--tooth', help='tooth number to predict', required=False, type=int, default=0)
    parser.add_argument('-r', '--rg', help='Radigraph to predict on', required=True, type=int)
    parser.add_argument('-g', '--rglocation', help='location of radigraphs', default='', type=str)
    parser.add_argument('-l', '--lmlocation', help='location of landmarks', default='', type=str)
    args = parser.parse_args()


    if args.tooth !=0:
        if len(args.rglocation) > 0 and len(args.lmlocation)>0:
            tlmk, Z, f1, otimg = main(args.tooth, args.rg, lmk_location=args.lmlocation, rg_location=args.rglocation)
        elif len(args.rglocation) > 0:
            tlmk, Z, f1, otimg = main(args.tooth, args.rg, rg_location=args.rglocation)
        elif len(args.lmlocation) > 0:
            tlmk, Z, f1, otimg = main(args.tooth, args.rg, lmk_location=args.lmlocation)
        else:
            tlmk, Z, f1, otimg = main(args.tooth, args.rg)
        draw_lmks_on_original([tlmk, Z], otimg, [(255,0,0), (0,255,0)])
    else:
        testlmks = []
        predlmks = []
        f1s = []
        for ttooth in [1,2,3,4,5,6,7,8]:
            if len(args.rglocation) > 0 and len(args.lmlocation)>0:
                tlmk, Z, f1, otimg = main(ttooth, args.rg, lmk_location=args.lmlocation, rg_location=args.rglocation)
            elif len(args.rglocation) > 0:
                tlmk, Z, f1, otimg = main(ttooth, args.rg, rg_location=args.rglocation)
            elif len(args.lmlocation) > 0:
                tlmk, Z, f1, otimg = main(ttooth, args.rg, lmk_location=args.lmlocation)
            else:
                tlmk, Z, f1, otimg = main(ttooth, args.rg)
            testlmks.append(tlmk)
            predlmks.append(Z)
            f1s.append(f1)
        with open('f1results.csv', 'a') as fw:
            csvw = csv.writer(fw, delimiter=';')
            for did in range(len(f1s)):
                line = [args.rg, did + 1, f1s[did]]
                csvw.writerow(line)
        draw_lmks_on_original(testlmks + predlmks, otimg, [(255,0,0)] * len(testlmks) + [(0,255,0)] * len(predlmks),
                              thickness=3, save=True, name='fitting_RG{}.png'.format(args.rg))