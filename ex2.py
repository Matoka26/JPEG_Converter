import numpy as np
from scipy import misc, ndimage
from utils import jpeg_decode, jpeg_encode, image_MSE
import matplotlib.pyplot as plt


if __name__ == "__main__":
    Y = X = misc.face()

    Q_luminance = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 28, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

    Q_chrominance = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                     [18, 21, 26, 66, 99, 99, 99, 99],
                     [24, 26, 56, 99, 99, 99, 99, 99],
                     [47, 66, 99, 99, 99, 99, 99, 99],
                     [99, 99, 99, 99, 99, 99, 99, 99],
                     [99, 99, 99, 99, 99, 99, 99, 99],
                     [99, 99, 99, 99, 99, 99, 99, 99],
                     [99, 99, 99, 99, 99, 99, 99, 99]])

    X = jpeg_encode(X, Q_luminance, Q_chrominance, MSE_trashold=300)
    X = jpeg_decode(X, Q_luminance, Q_chrominance, 1)