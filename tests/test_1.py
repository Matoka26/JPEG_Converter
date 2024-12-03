import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

from utils import jpeg_mono_decode, jpeg_mono_encode

if __name__ == "__main__":
    X = misc.ascent()
    n, m = X.shape
    tests = [(n, m), (n-300, m-300), (n, m-69), (n-315, m-500)]

    Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

    for i, t in enumerate(tests):
        print(f'For shape {t[0]}*{t[1]}')
        Y_jpeg = jpeg_mono_encode(X[:t[0], :t[1]], Q_jpeg, print_freq_comps_cnt=True)
        X_jpeg = jpeg_mono_decode(Y_jpeg, filter_shape=Q_jpeg.shape, show_compressed_img=True)
        plt.imshow(X_jpeg)
        plt.title(f'Shape {X_jpeg.shape}')
        plt.savefig(f"./outputs/test1_{i}.pdf")

