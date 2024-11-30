import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn


def jpeg_decode_sliding_window(Y: np.ndarray, Q_jpeg: np.ndarray, show_compressed_img=0) -> np.ndarray:
    i_lim, j_lim = Y.shape
    i_win, j_win = Q_jpeg.shape
    X_jpeg = np.zeros(Y.shape)

    # decode each block
    for i in range(0, i_lim, i_win):
        for j in range(0, j_lim, j_win):

            # exctract window
            y_slice = Y[i:i + i_win, j:j + j_win]

            try:
            # decode
                X_jpeg[i:i + i_win, j:j + j_win] = idctn(y_slice)
            except ValueError:
                X_jpeg[i:i + y_slice.shape[0], j:j + y_slice.shape[1]] = idctn(y_slice)

    if show_compressed_img:
        plt.imshow(X_jpeg, cmap=plt.cm.gray)
        plt.show()

    return X_jpeg


def jpeg_encode_sliding_window(X: np.ndarray, Q_jpeg: np.ndarray, print_freq_comps_cnt=0) -> np.ndarray:
    i_lim, j_lim = X.shape
    i_win, j_win = Q_jpeg.shape
    Y_jpeg = np.zeros(X.shape)

    for i in range(0, i_lim, i_win):
        for j in range(0, j_lim, j_win):

            # extract window
            x_slice = X[i:i+i_win, j:j+j_win]

            # encode
            try:
                y = dctn(x_slice)
                Y_jpeg[i:i+i_win, j:j+j_win] = Q_jpeg * np.round(y/Q_jpeg)
            except ValueError:
                # if Q mismatches x_slice, slice from Q
                # alternatively, you could pad x with zeros
                q_segment = Q_jpeg[:x_slice.shape[0], :x_slice.shape[1]]
                Y_jpeg[i:i+i_win, j:j+j_win] = q_segment * np.round(y/q_segment)

    # print non-zero components
    if print_freq_comps_cnt:
        y_nnz = np.count_nonzero(dctn(X))
        y_jpeg_nnz = np.count_nonzero(Y_jpeg)
        print('Componente în frecvență:' + str(y_nnz) +
              '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))

    # return coded jpeg
    return Y_jpeg


if __name__ == "__main__":
    X = misc.ascent()
    n, m = X.shape
    tests = [(n, m), (n-300, m-300), (n, m-69)]

    Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

    for t in tests:
        print(f'For shape {t[0]}*{t[1]}')
        Y_jpeg = jpeg_encode_sliding_window(X[:t[0], :t[1]], Q_jpeg, 1)
        X_jpeg = jpeg_decode_sliding_window(Y_jpeg, Q_jpeg, 1)
