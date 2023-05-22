import cv2
import numpy as np
import skimage
from scipy.ndimage import gaussian_filter, maximum_filter


def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    def cal_ix_iy(image: np.array, ksize):
        img = np.copy(image)
        img = skimage.img_as_float32(img)
        ix = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize)
        iy = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize)
        return ix, iy

    def get_r(ix, iy, ksize, k, width):
        # This function applies the window (the Gaussian filter to the derivatives then calculates R value)
        I_x = cv2.GaussianBlur(ix ** 2, (ksize, ksize), 0)
        I_y = cv2.GaussianBlur(iy ** 2, (ksize, ksize), 0)
        H_xy = cv2.GaussianBlur(ix * iy, (ksize, ksize), 0)
        R = ((I_x * I_y - np.square(H_xy)) - (k * np.square(I_x + I_y)))

        return R

    sobel_ksize = 5
    if (feature_width % 2 == 0):
        gaussian_ksize = feature_width + 1
    else:
        gaussian_ksize = feature_width + 1
    k = 0.0001

    ix, iy = cal_ix_iy(image, sobel_ksize)

    R = get_r(ix, iy, gaussian_ksize, k, 16)

    # Get the coordinates of the interest points
    coordinates = maximum_filter(R, size=3)

    x = []
    y = []
    for i in range(coordinates.shape[0]):
        for j in range(coordinates.shape[1]):
            if coordinates[i][j] != 0:
                x.append(j)
                y.append(i)

    return np.array(x), np.array(y)


def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    feature = np.zeros((16, 8))
    all_features = np.zeros((len(x), 128))

    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    gauss_img = gaussian_filter(image, sigma=1.5)

    dx, dy = np.gradient(gauss_img)

    grad = np.sqrt(np.square(dx) + np.square(dy))
    img_orientation = np.arctan2(dy, dx) * (180 / np.pi)
    img_orientation[img_orientation < 0] += (2 * np.pi)

    for i in range(0, 8):
        x_, y_ = np.where(((img_orientation >= (i * 45)) &
                          (img_orientation < ((i + 1) * 45))))
        img_orientation[x_, y_] = int(i + 1)

    window_size = np.round(feature_width / 4).astype(int)

    for i in range(len(x)):
        temp_orientation = img_orientation[y[i] -
                                           8:y[i] + 8, x[i] - 8:x[i] + 8]
        temp_grad = grad[y[i] - 8:y[i] + 8, x[i] - 8:x[i] + 8]

        x_dir = 0
        y_dir = 0

        for j in range(16):
            window = temp_orientation[y_dir:y_dir +
                                      window_size, x_dir:x_dir + window_size]
            grad_win = temp_grad[y_dir:y_dir +
                                 window_size, x_dir:x_dir + window_size]

            hist = np.histogram(window, bins=8, range=(1, 9), weights=grad_win)
            feature[j, :] = np.array(hist[0])
            y_dir += 4

            if y_dir == feature_width:
                x_dir += 4
                y_dir = 0

        feature_reshaped = feature.reshape(1, -1)
        all_features[i, :] = feature_reshaped

    norm = np.linalg.norm(all_features, axis=1).reshape(-1, 1)
    norm[norm == 0] = 0.01
    all_features = all_features / norm

    all_features[all_features > 0.2] = 0.2

    norm2 = np.linalg.norm(all_features, axis=1).reshape(-1, 1)
    norm2[norm2 == 0] = 0.01
    features = all_features / norm2

    return features ** 0.75


def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    matches = []
    confidences = []

    for i in range(im1_features.shape[0]):
        euclidean_dist = np.sqrt(
            ((im1_features[i, :] - im2_features) ** 2).sum(axis=1))
        sorted_dist = np.argsort(euclidean_dist)

        if (euclidean_dist[sorted_dist[0]] / euclidean_dist[sorted_dist[1]]) < 0.93:
            matches.append([i, sorted_dist[0]])
            confidences.append(
                (1.0 - euclidean_dist[sorted_dist[0]] / euclidean_dist[sorted_dist[1]]) * 100)

    return np.asarray(matches), np.asarray(confidences)
