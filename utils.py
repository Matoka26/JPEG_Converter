import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn


def jpeg_mono_decode(Y: np.ndarray, Q_jpeg: np.ndarray, show_compressed_img=0) -> np.ndarray:
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


def jpeg_mono_encode(X: np.ndarray, Q_jpeg: np.ndarray, print_freq_comps_cnt=0) -> np.ndarray:
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


def rgb_to_ycbcr(X: np.ndarray) -> np.ndarray:

    Q = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])

    b = np.array([0, 128, 128])

    X_ycbcr = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_ycbcr[i, j] = np.dot(Q, X[i, j]) + b

    return np.clip(X_ycbcr, 0, 255).astype(np.uint8)


def ycbcr_to_rgb(X:np.ndarray) -> np.ndarray:
    Q = np.array([[1, 0, 1.402],
                  [1, -0.344136, -0.714136],
                  [1, 1.772, 0]])

    b = np.array([0, 128, 128])

    X_rgb = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_rgb[i, j] = np.dot(Q, (X[i, j] - b))

    return np.clip(X_rgb, 0, 255).astype(np.uint8)


def jpeg_encode(X: np.ndarray, Q_Y: np.ndarray, Q_Ch: np.ndarray, print_freq_comps_cnt=0) -> np.ndarray:

    # color space conversion
    X_ycbcr = rgb_to_ycbcr(X)

    # split channels
    Y =  X_ycbcr[:, :, 0]
    Cb = X_ycbcr[:, :, 1]
    Cr = X_ycbcr[:, :, 2]

    # encode channels
    Y =  jpeg_mono_encode(Y, Q_Y)
    Cb = jpeg_mono_encode(Cb, Q_Ch)
    Cr = jpeg_mono_encode(Cr, Q_Ch)

    # reconstruct image
    X_jpeg = np.stack((Y, Cb, Cr), axis=-1)

    # print non-zero components
    if print_freq_comps_cnt:
        y_nnz = np.count_nonzero(dctn(X))
        y_jpeg_nnz = np.count_nonzero(X_jpeg)
        print('Componente în frecvență:' + str(y_nnz) +
              '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))

    return X_jpeg


def jpeg_decode(X: np.ndarray, Q_Y: np.ndarray, Q_Ch: np.ndarray, show_compressed_img=0) -> np.ndarray:

    # split channels
    Y =  X[:, :, 0]
    Cb = X[:, :, 1]
    Cr = X[:, :, 2]

    # decode channels
    Y = jpeg_mono_decode(Y, Q_Y)
    Cb = jpeg_mono_decode(Cb, Q_Ch)
    Cr = jpeg_mono_decode(Cr, Q_Ch)

    # reconstruct image
    X_jpeg = np.stack((Y, Cb, Cr), axis=-1)

    # color space conversion
    X_jpeg = ycbcr_to_rgb(X_jpeg)

    if show_compressed_img:
        plt.imshow(X_jpeg)
        plt.show()

    return X_jpeg
