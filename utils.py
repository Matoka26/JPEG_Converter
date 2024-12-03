import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from sklearn.metrics import mean_squared_error

# Standard Quantization table for Luminance Spectrum
Q_luminance = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                        [12, 12, 14, 19, 26, 28, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56],
                        [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77],
                        [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103, 99]])

# Standard Quantization table for Chrominance Spectrums
Q_chrominance = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                          [18, 21, 26, 66, 99, 99, 99, 99],
                          [24, 26, 56, 99, 99, 99, 99, 99],
                          [47, 66, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99]])


def jpeg_mono_decode(Y: np.ndarray, filter_shape: (int,int), show_compressed_img: bool=False) -> np.ndarray:
    # Decode momo-chromatic channel using the size of the filter previously applied

    i_lim, j_lim = Y.shape
    i_win, j_win = filter_shape
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


def jpeg_mono_encode(X: np.ndarray, Q_jpeg: np.ndarray, print_freq_comps_cnt: bool=False) -> np.ndarray:
    # Encode mono-chormatic channel using a quantization table

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
    # Convert RGB to YCbCr Spectrum

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
    # Convert YCbCr to RGB Spectrum

    Q = np.array([[1, 0, 1.402],
                  [1, -0.344136, -0.714136],
                  [1, 1.772, 0]])

    b = np.array([0, 128, 128])

    X_rgb = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_rgb[i, j] = np.dot(Q, (X[i, j] - b))

    return np.clip(X_rgb, 0, 255).astype(np.uint8)


def jpeg_encode(X: np.ndarray = Q_luminance, Q_Y: np.ndarray = Q_chrominance, Q_Ch: np.ndarray = Q_chrominance
                , MSE_trashold: float=None , print_freq_comps_cnt:bool=0) -> np.ndarray:
    # Encode RGB image

    mse_coef = 1

    # color space conversion
    X_ycbcr = rgb_to_ycbcr(X)

    # split channels
    Y =  X_ycbcr[:, :, 0]
    Cb = X_ycbcr[:, :, 1]
    Cr = X_ycbcr[:, :, 2]

    prev_mse = -1
    while(True):
        # encode channels
        Y_enc =  jpeg_mono_encode(Y, Q_Y * mse_coef)
        Cb_enc = jpeg_mono_encode(Cb, Q_Ch * mse_coef)
        Cr_enc = jpeg_mono_encode(Cr, Q_Ch * mse_coef)

        # reconstruct image
        X_jpeg = np.stack((Y_enc, Cb_enc, Cr_enc), axis=-1)

        # exit if it is not specified
        if MSE_trashold is None:
            break

        # get MSE for decoded iamge
        X_decoded = jpeg_decode(X_jpeg, Q_Y.shape, Q_Ch.shape)
        currect_mse = image_MSE(X, X_decoded)

        if prev_mse == currect_mse:
            raise ValueError("Couldnt reach MSE thrashold")

        # if it touches the thrashold, exit
        tolerance = 0.3
        if abs(currect_mse - MSE_trashold) < tolerance:
            break

        # update quantization matrices coeficient
        if currect_mse < MSE_trashold:
            mse_coef *= 1.401
        else:
            mse_coef /= 1.219
        prev_mse = currect_mse

    # print non-zero components
    if print_freq_comps_cnt:
        y_nnz = np.count_nonzero(dctn(X))
        y_jpeg_nnz = np.count_nonzero(X_jpeg)
        print('Componente în frecvență:' + str(y_nnz) +
              '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))

    return X_jpeg


def jpeg_decode(X: np.ndarray = Q_luminance, Q_Y_shape: (int,int)=Q_chrominance.shape, Q_Ch_shape: (int,int)=Q_chrominance.shape
                , show_compressed_img=0) -> np.ndarray:
    # Decode YCbCr image

    # split channels
    Y =  X[:, :, 0]
    Cb = X[:, :, 1]
    Cr = X[:, :, 2]

    # decode channels
    Y = jpeg_mono_decode(Y, Q_Y_shape)
    Cb = jpeg_mono_decode(Cb, Q_Ch_shape)
    Cr = jpeg_mono_decode(Cr, Q_Ch_shape)

    # reconstruct image
    X_jpeg = np.stack((Y, Cb, Cr), axis=-1)

    # color space conversion
    X_jpeg = ycbcr_to_rgb(X_jpeg)

    if show_compressed_img:
        plt.imshow(X_jpeg)
        plt.show()

    return X_jpeg


def image_MSE(X: np.ndarray, Y: np.ndarray) -> float:
    # Get Mean Squared Error of 2 images
    return mean_squared_error(X.flatten(), Y.flatten())


def display_video(video_path: str, window_name: str='Video', frame_delay:int =25):
    # Play video

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow(window_name, frame)
        if cv.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def jpeg_video_encode(video_path: str, output_video: str) -> None:
    # Encode each frame of a video

    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read the first frame.")
        return

    video = cv.VideoWriter(output_video, cv.VideoWriter_fourcc(*'mp4v'), fps, frame.shape[:2][::-1])
    video.write(jpeg_encode(frame))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video.write(frame)

    cap.release()
    video.release()
    cv.destroyAllWindows()
