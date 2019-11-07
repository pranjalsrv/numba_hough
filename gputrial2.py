from numba import cuda
import math
import numpy as np
import cv2
import time


@cuda.jit
def create_hough_space_with_acc(hough_space, img):
    p, q = cuda.grid(2)
    #i = cuda.grid(1)

    if p < hough_space.shape[0] and q < hough_space.shape[1]:

        for x in range(x_max):
            for y in range(y_max):
                if img[x, y] == 255:
                    continue
                for itheta in range(theta_dim):
                    theta = 1.0 * itheta * theta_max / theta_dim
                    r = x * math.cos(theta) + y * math.sin(theta)
                    ir = r_dim * (1.0 * r) / r_max
                    hough_space[int(ir), int(itheta)] = hough_space[int(ir), int(itheta)] + 1

    # if p < hough_space.shape[0] / 2 and q < hough_space.shape[1] / 2:
    #     for x in range(x_max / 2):
    #         for y in range(y_max / 2):
    #             if img[x, y] == 255:
    #                 continue
    #             for itheta in range(theta_dim):
    #                 theta = 1.0 * itheta * theta_max / theta_dim
    #                 r = x * math.cos(theta) + y * math.sin(theta)
    #                 ir = r_dim * (1.0 * r) / r_max
    #                 hough_space[int(ir), int(itheta)] = hough_space[int(ir), int(itheta)] + 1
    #
    # elif p < hough_space.shape[0] / 2 and q > hough_space.shape[1] / 2:
    #     for x in range(x_max / 2):
    #         for y in range(y_max / 2 + 1, y_max):
    #             if img[x, y] == 255:
    #                 continue
    #             for itheta in range(theta_dim):
    #                 theta = 1.0 * itheta * theta_max / theta_dim
    #                 r = x * math.cos(theta) + y * math.sin(theta)
    #                 ir = r_dim * (1.0 * r) / r_max
    #                 hough_space[int(ir), int(itheta)] = hough_space[int(ir), int(itheta)] + 1
    #
    # elif p > hough_space.shape[0] / 2 and q < hough_space.shape[1] / 2:
    #     for x in range(x_max / 2 + 1, x_max):
    #         for y in range(y_max / 2):
    #             if img[x, y] == 255:
    #                 continue
    #             for itheta in range(theta_dim):
    #                 theta = 1.0 * itheta * theta_max / theta_dim
    #                 r = x * math.cos(theta) + y * math.sin(theta)
    #                 ir = r_dim * (1.0 * r) / r_max
    #                 hough_space[int(ir), int(itheta)] = hough_space[int(ir), int(itheta)] + 1
    #
    # elif p > hough_space.shape[0] / 2 and q > hough_space.shape[1] / 2:
    #     for x in range(x_max / 2 + 1, x_max):
    #         for y in range(y_max / 2, y_max):
    #             if img[x, y] == 255:
    #                 continue
    #             for itheta in range(theta_dim):
    #                 theta = 1.0 * itheta * theta_max / theta_dim
    #                 r = x * math.cos(theta) + y * math.sin(theta)
    #                 ir = r_dim * (1.0 * r) / r_max
    #                 hough_space[int(ir), int(itheta)] = hough_space[int(ir), int(itheta)] + 1


r_dim = 200
theta_dim = 300
theta_max = 1.0 * math.pi
img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x_max, y_max = img.shape
r_max = math.hypot(x_max, y_max)
hough_space = np.zeros((r_dim, theta_dim))

threadsperblock = (16, 16)
# blockspergrid_x = 32
# blockspergrid_y = 32

blockspergrid_x = math.ceil(hough_space.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(hough_space.shape[1] / threadsperblock[1])

blockspergrid = (blockspergrid_x, blockspergrid_y)

dev_hough_space = cuda.to_device(hough_space)
dev_img = cuda.to_device(img)
# x_max
# y_max
# img
# r_max
# r_dim

# blockspergrid, threadsperblock
start = time.time()
create_hough_space_with_acc[blockspergrid, threadsperblock](dev_hough_space, dev_img)
end = time.time()

# cuda.synchronize()
#arr = dev_hough_space.copy_to_host()
# ary = np.empty(shape=dev_hough_space.shape, dtype=dev_hough_space.dtype)
# dev_hough_space.copy_to_host(ary)

#print(arr)

print(end - start)

# cuda.synchronize()
