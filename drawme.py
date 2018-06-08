import cv2
import numpy as np
import colorsys
from utils import scale_by_param
from utils import scale_to_unit
from utils import flatten_special, unflatten_special
SCREEN_H = 800
SCREEN_W = 1200
from utils import get_center
def translate_by_vec(lmk, trvec):
    return lmk + trvec


def fit_to_screen(image):
    scale = min(float(SCREEN_W) / image.shape[1], float(SCREEN_H) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))


def __get_colors(num_colors):
    """
        http://stackoverflow.com/a/9701141
    """
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in colors]


def draw_gpa(mean_lmk, align_lmks):
    img = np.ones((1000, 600, 3), np.uint8) * 255

    # plot mean shape
    mean_lmk = scale_by_param(mean_lmk, 1500)
    mean_lmk = translate_by_vec(mean_lmk, [300, 500])

    points = mean_lmk
    for i in range(len(points)):
        cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                 (int(points[(i + 1) % 40, 0]), int(points[(i + 1) % 40, 1])),
                 (0, 0, 0), 2)
    ## center of mean shape
    cv2.circle(img, (300, 500), 10, (255, 255, 255))

    # plot aligned shapes
    colors = __get_colors(len(align_lmks))
    for ind, aligned_shape in enumerate(align_lmks):
        #aligned_shape = aligned_shape.scale(1500).translate([300, 500])
        aligned_shape = scale_by_param(aligned_shape, 1500)
        aligned_shape = translate_by_vec(aligned_shape, [300, 500])
        points = aligned_shape
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[(i + 1) % 40, 0]), int(points[(i + 1) % 40, 1])),
                     colors[ind])

    # show
    img = fit_to_screen(img)
    cv2.imshow('Procrustes result for incisor ' + str(1111), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_asm(asm, incisor_nr=0, save=False):
    __plot_mode(flatten_special(asm.mean_shape), asm.pc_modes[:, 0], title="PCA/incisor"+str(incisor_nr)+"mode1", save=save)
    __plot_mode(flatten_special(asm.mean_shape), asm.pc_modes[:, 1], title="PCA/incisor"+str(incisor_nr)+"mode2", save=save)
    __plot_mode(flatten_special(asm.mean_shape), asm.pc_modes[:, 2], title="PCA/incisor"+str(incisor_nr)+"mode3", save=save)
    __plot_mode(flatten_special(asm.mean_shape), asm.pc_modes[:, 3], title="PCA/incisor"+str(incisor_nr)+"mode4", save=save)


def __plot_mode(mu, pc, title="Active Shape Model", save=False):
    """Plot the mean shape +/- nstd times the principal component
    """
    colors = [(224, 224, 224), (160, 160, 160), (64, 64, 64), (0, 0, 0),
              (64, 64, 64), (160, 160, 160), (224, 224, 224)
             ]
    shapes = [unflatten_special(mu-3*pc),
              unflatten_special(mu-2*pc),
              unflatten_special(mu-1*pc),
              unflatten_special(mu),
              unflatten_special(mu+1*pc),
              unflatten_special(mu+2*pc),
              unflatten_special(mu+3*pc)
             ]
    plot_shapes(shapes, colors, title, save)


def plot_shapes(shapes, colors, title="Shape Model", save=False):
    """Function to show all of the shapes which are passed to it.
    """
    cv2.namedWindow(title)
    shapes = [scale_by_param(scale_to_unit(dshape), 1000) for dshape in shapes]
    #shapes = [shape.scale_to_unit().scale(1000) for shape in shapes]

    max_x = int(max([dshape[:, 0].max() for dshape in shapes]))
    max_y = int(max([dshape[:, 1].max() for dshape in shapes]))
    min_x = int(min([dshape[:, 0].min() for dshape in shapes]))
    min_y = int(min([dshape[:, 1].min() for dshape in shapes]))

    img = np.ones((max_y-min_y+20, max_x-min_x+20, 3), np.uint8)*255
    for shape_num, shape in enumerate(shapes):
        points = shape
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]-min_x+10), int(points[i, 1]-min_y+10)),
                     (int(points[(i + 1) % 40, 0]-min_x+10), int(points[(i + 1) % 40, 1]-min_y+10)),
                     colors[shape_num], thickness=1, lineType=8)

    cv2.imshow(title, img)
    cv2.waitKey()
    if save:
        cv2.imwrite('Plot/'+title+'.png', img)
    cv2.destroyAllWindows()



def draw_lmks_on_original(lms_list, img, colors, save=False, name=None, thickness=1):
    img = img.copy()


    for ind, lms in enumerate(lms_list):
        points = lms
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[(i + 1)%40, 0]), int(points[(i + 1)%40, 1])),
                     colors[ind], thickness)


    img = fit_to_screen(img)
    if save:
        cv2.imwrite(name, img)
    else:
        cv2.imshow('landmarks on image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

