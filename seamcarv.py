
import numpy
__author__ = 'Anant'

from pylab import *
from skimage import img_as_float, color
from skimage.filters import sobel_h, sobel_v

inf = 1e1000


def dual_gradient_energy(img):
    """
    Calculating Energy gradient
    :return 3D image matrix,
    the returned matrix is 3D to enable plotting.
    """
    R_sobel_h = sobel_h(img[:, :, 0])
    R_sobel_v = sobel_v(img[:, :, 0])
    G_sobel_h = sobel_h(img[:, :, 1])
    G_sobel_v = sobel_v(img[:, :, 1])
    B_sobel_h = sobel_h(img[:, :, 2])
    B_sobel_v = sobel_v(img[:, :, 2])
    a = img[:, 0, 0].size
    b = img[0, :, 0].size
    energy = numpy.zeros((a, b, 3))
    sob = numpy.zeros((a, b, 3))
    for i in range(0, img[:, 0, 0].size):
        for j in range(0, img[0, :, 0].size):
            energy[i, j, 0] = R_sobel_h[i, j]**2 + R_sobel_v[i, j]**2
            energy[i, j, 1] = G_sobel_h[i, j]**2 + G_sobel_v[i, j]**2
            energy[i, j, 2] = B_sobel_h[i, j]**2 + B_sobel_v[i, j]**2
            sob[i, j, :] = energy[i, j, 0] + energy[i, j, 1] + energy[i, j, 2]
    return sob


def find_horizontal_seam(im):

    """
    Takes a grayscale img and returns the lowest
    energy horizontal seam as a list of pixels (2-tuples).
    This implements the dynamic programming seam-find
    algorithm. For an m*n picture, this algorithm
    takes O(m*n) time
    @im: a grayscale image
    :return path of the least energy seam.
    """

    im_height, im_width = im.shape
    seam_dir = np.zeros((im_height, im_width))
    energy_seam = np.zeros((im_height, im_width))
    energy_seam[1, :] = im[1, :]
    for i in range(2, im_height-1):
        for j in range(1, im_width-1):
            if j == 1:
                m = min(energy_seam[i-1, j], energy_seam[i-1, j+1])
                energy_seam[i, j] = m + im[i, j]
                if m == energy_seam[i-1, j]:
                    seam_dir[i, j] = 1
                else:
                    seam_dir[i, j] = 2
            elif j == im_width-2:
                m = min(energy_seam[i-1, j-1], energy_seam[i-1, j])
                energy_seam[i, j] = m + im[i, j]
                if m == energy_seam[i-1, j]:
                    seam_dir[i, j] = 1
                else:
                    seam_dir[i, j] = 0
            else:
                m = min(energy_seam[i-1, j-1],
                        energy_seam[i-1, j], energy_seam[i-1, j+1])
                energy_seam[i, j] = m + im[i, j]
                if m == energy_seam[i-1, j-1]:
                    seam_dir[i, j] = 0
                elif m == energy_seam[i-1, j]:
                    seam_dir[i, j] = 1
                else:
                    seam_dir[i, j] = 2

    # print energy_seam, seam_dir

    """
    find minimum value of seam energy
    """
    min_val = inf
    for k in range(1, im_width-1):
        if energy_seam[im_height-2, k] < min_val:
            min_val = min(min_val, energy_seam[im_height-2, k])
            x = k
        elif energy_seam[im_height-2, k] == min_val:
            x = k
    # print x

    """
    find the path:
    """
    path = []
    for i in range(im_height-2, 0, -1):
        # print i,x
        pos = (i, x)
        path.append(pos)
        if seam_dir[i, x] == 0:
            x -= 1
        if seam_dir[i, x] == 2:
            x += 1
    return path


def plot_seam(path, energy):
    """
    plots the minimum seam.
    returns a new image which shows the plot of the minimum
    seam on the energy map.
    Displays plot.
    """
    for i, j in path:
        energy[i, j, :] = [1, 0, 0]
    gray()
    imshow(energy)
    show()


def delete_horizontal_seam(img, path):
    """
    Delete the path in the original image.
    returns a new image which is W-1xHx3
    """
    im_height, im_width, arr = img.shape
    temp = np.zeros((im_height, im_width-1, arr))

    for i in range(im_height):
        flag = False
        for j in range(im_width-1):
            if (i, j) not in path and flag is False:
                temp[i, j, :] = img[i, j, :]
            elif (i, j) in path:
                flag = True
                temp[i, j, :] = img[i, j+1, :]
            elif (i, j) not in path and flag is True:
                temp[i, j, :] = img[i, j+1, :]
    return temp


def main():
    """
    mask is the pixel size to reduce,
    :default 1
    otuput.png size W-1xHx3
    """
    mask = 1
    img = imread('tower.png')
    img = numpy.array(img_as_float(img))
    # print img.shape
    for i in range(mask):
        energy = dual_gradient_energy(img)
        # energy = np.array([[0,0,0,0,0,0,0],[0,1,4,3,5,2,0],
        # [0,3,2,5,2,3,0],[0,5,2,4,2,1,0],[0,0,0,0,0,0,0]]
        path = find_horizontal_seam(color.rgb2gray(energy))
        img = delete_horizontal_seam(img, path)
    plot_seam(path, energy)
    imsave('output', img)


if __name__ == "__main__":
    main()
