import numpy as np
from scipy import misc, ndimage
from utils import jpeg_decode, jpeg_encode, image_MSE
import matplotlib.pyplot as plt


if __name__ == "__main__":

    tests = [None, 32, 12.1, 100, 3]

    for i, t in enumerate(tests):
        X = misc.face()
        X = jpeg_encode(X, MSE_trashold=t)
        X = jpeg_decode(X, show_compressed_img=True)
        plt.imshow(X)
        plt.title(f"MSE {t}")
        plt.savefig(f"./outputs/test2_{i}.pdf")

