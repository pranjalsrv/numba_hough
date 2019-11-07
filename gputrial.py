from numba import jit, prange, cuda
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import time

# ---------------------------------------------------------------------------------------- #
# Step 1: read image


r_dim = 200
theta_dim = 300
theta_max = 1.0 * math.pi
outer_hough_space = np.zeros((r_dim, theta_dim))


def read_img(img_path):
    img = cv2.imread(img_path)  # pentagon.png
    # print('image shape: ', img.shape)
    # plt.imshow(img, )
    # plt.savefig("image1.png", bbox_inches='tight')
    # plt.close()

    return img


# ----------------------------------------------------------------------------------------#
# Step 2: Hough Space


# @jit(nopython=True, parallel=True, fastmath=True)
@cuda.jit
def create_hough_space_with_acc(img, hough_space, r_max):
    # hough_space = np.zeros((r_dim, theta_dim))

    i = cuda.grid(1)

    if i < hough_space.shape[0]:
        for x in range(x_max):
            for y in range(y_max):
                if img[x, y] == 255:
                    continue
                for itheta in range(theta_dim):
                    theta = 1.0 * itheta * theta_max / theta_dim
                    r = x * math.cos(theta) + y * math.sin(theta)
                    ir = r_dim * (1.0 * r) / r_max
                    hough_space[int(ir), int(itheta)] = hough_space[int(ir), int(itheta)] + 1

        #outer_hough_space.copy_to_host(hough_space)


def create_hough_space_without_acc(img):
    x_max, y_max = img.shapea

    theta_max = 1.0 * math.pi
    r_max = math.hypot(x_max, y_max)

    hough_space = np.zeros((r_dim, theta_dim))

    for x in range(x_max):
        for y in range(y_max):
            if img[x, y] == 255:
                continue
            for itheta in range(theta_dim):
                theta = 1.0 * itheta * theta_max / theta_dim
                r = x * math.cos(theta) + y * math.sin(theta)
                ir = r_dim * (1.0 * r) / r_max
                hough_space[int(ir), int(itheta)] = hough_space[int(ir), int(itheta)] + 1

    return hough_space, r_dim, r_max, theta_dim, theta_max, y_max


# Step 3: Find maximas 2

def find_maxima(hough_space):
    neighborhood_size = 20
    threshold = 140

    data_max = filters.maximum_filter(hough_space, neighborhood_size)
    maxima = (hough_space == data_max)

    data_min = filters.minimum_filter(hough_space, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

    return x, y

    # plt.imshow(hough_space, origin='lower')
    # plt.savefig('hough_space_i_j.png', bbox_inches='tight')
    #
    # plt.autoscale(False)
    # plt.plot(x, y, 'ro')
    # plt.savefig('hough_space_maximas.png', bbox_inches='tight')
    #
    # plt.close()


# ----------------------------------------------------------------------------------------#
# Step 4: Plot lines


def hough_transform_opencv(gray, output_path, img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=5)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    print(lines.shape)
    for i in lines:
        for rho, theta in i:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(output_path, img)


def plot_line(x, y, r_max, y_max, img, output_path):
    line_index = 1

    fig, ax = plt.subplots()
    ax.imshow(img)

    for i, j in zip(y, x):

        r = round((1.0 * i * r_max) / r_dim, 1)
        theta = round((1.0 * j * theta_max) / theta_dim, 1)

        ax.autoscale(False)

        px = []
        py = []
        for i in range(-y_max - 40, y_max + 40, 1):
            px.append(math.cos(-theta) * i - math.sin(-theta) * r)
            py.append(math.sin(-theta) * i + math.cos(-theta) * r)

        ax.plot(px, py, linewidth=4)

        line_index = line_index + 1

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    img = read_img('image.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hough_space = np.zeros((r_dim, theta_dim))
    x_max, y_max = gray.shape
    r_max = math.hypot(x_max, y_max)
    # dev_hough_space = cuda.to_device(hough_space)
    # dev_gray = cuda.to_device(gray)
    # dev_r_max = cuda.to_device(r_max)

    print("\nWITH JIT COMPILER ACCELERATION:\n")
    print('\nFirst Iteration:')
    start = time.time()
    # threadsperblock = 32
    # #print(img.size)
    # blockspergrid = (img.shape[0] + (threadsperblock - 1)) // threadsperblock
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(hough_space.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(hough_space.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    create_hough_space_with_acc[blockspergrid, threadsperblock](hough_space)
    #result = dev_hough_space.copy_to_host()
    # hough_space, r_max, y_max = create_hough_space_with_acc(gray)
    # hough_end = time.time()
    # x, y = find_maxima(hough_space)
    maxima_end = time.time()
    # plot_line(x, y, r_max, y_max, img, "images/acc_it1.png")

    #
    # # print('time taken for hough space: ', (hough_end - start))
    # # print('time taken for finding maxima: ', (maxima_end - hough_end))
    print('overall time: ', (maxima_end - start))
    #
    # print('\nSecond Iteration:')
    # start = time.time()
    # hough_space, r_max, y_max = create_hough_space_with_acc(gray)
    # # hough_end = time.time()
    # x, y = find_maxima(hough_space)
    # maxima_end = time.time()
    # plot_line(x, y, r_max, r_dim, theta_dim, theta_max, y_max, img, "images/acc_it2.png")
    #
    # # print('time taken for hough space: ', (hough_end - start))
    # # print('time taken for finding maxima: ', (maxima_end - hough_end))
    # print('overall time: ', (maxima_end - start))
    #
    # acc_time = maxima_end - start
    #
    # print("\n\nWITHOUT JIT COMPILER ACCELERATION:\n")
    # print('\nFirst Iteration:')
    # start = time.time()
    # hough_space, r_dim, r_max, theta_dim, theta_max, y_max = create_hough_space_without_acc(gray)
    # # hough_end = time.time()
    # x, y = find_maxima(hough_space)
    # maxima_end = time.time()
    # plot_line(x, y, r_max, r_dim, theta_dim, theta_max, y_max, img, "images/normal_it1.png")
    #
    # # print('time taken for hough space: ', (hough_end - start))
    # # print('time taken for finding maxima: ', (maxima_end - hough_end))
    # print('overall time: ', (maxima_end - start))
    #
    # print('\nSecond Iteration:')
    # start = time.time()
    # hough_space, r_dim, r_max, theta_dim, theta_max, y_max = create_hough_space_without_acc(gray)
    # # hough_end = time.time()
    # x, y = find_maxima(hough_space)
    # maxima_end = time.time()
    # plot_line(x, y, r_max, r_dim, theta_dim, theta_max, y_max, img, "images/normal_it2.png")
    #
    # # print('time taken for hough space: ', (hough_end - start))
    # # print('time taken for finding maxima: ', (maxima_end - hough_end))
    # print('overall time: ', (maxima_end - start))
    # without_acc_time = maxima_end - start
    #
    # print('\n\nSPEEDUP = ', (without_acc_time / acc_time))
    #
    # print("\n\nOpenCV Output:\n")
    #
    # print("\nFirst Iteration")
    # start = time.time()
    # hough_transform_opencv(gray, "images/ocv_it1.png", img)
    # end = time.time()
    # print("Time taken: ", (end - start))
    #
    # print("\nSecond Iteration")
    # start = time.time()
    # hough_transform_opencv(gray, "images/ocv_it2.png", img)
    # end = time.time()
    # print("Time taken: ", (end - start))
