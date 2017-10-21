import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.exposure import equalize_adapthist

from scipy.ndimage.measurements import label


def load_img(imgpath):
    return plt.imread(imgpath)


def equalize_CLAHE(img):
    img = np.uint8(img * 255)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(4, 4))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:, :, 2] = clahe.apply(img[:, :, 2])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def img_normalize(img):
    return (img - img.min())/(img.max()-img.min())


def image_preprocess(img):
    """Apply preprocessing.

    1. Apply gaussian blur
    2. Apply CLAHE equalization

    Args
    -------
    img:  target image

    Returns
    ----------
    img_out: image applied preprocessing

    """

    #img = np.copy(img)
    #g_kernel_size = 5
    #img = cv2.GaussianBlur(img, (g_kernel_size, g_kernel_size), 0)
    #img = img_normalize(img)
    #img = equalize_adapthist(img, clip_limit=0.01, kernel_size=32)
    #img = equalize_CLAHE(img)

    return img


def img_augmentation(img, n_sample, is_vehicle=True):
    import skimage.transform
    import scipy as sp

    def get_truncnorm(a, b, mu, sigma): return sp.stats.truncnorm(
        (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

    if is_vehicle:
        angle_rng = get_truncnorm(-10.0, 10.0, 0.0, 5.0)
        scale_rng = get_truncnorm(0.8, 1.2, 1.0, 0.2)
        shear_rng = get_truncnorm(-0.2, 0.2, 0.0, 0.1)
        trans_rng = get_truncnorm(0.1, -0.1, 0.0, 0.01)
    else:
        angle_rng = get_truncnorm(-1.0, 00.0, 0.0, 5.0)
        scale_rng = get_truncnorm(0.8, 1.2, 1.0, 0.2)
        shear_rng = get_truncnorm(-0.5, 0.5, 0.0, 0.1)
        trans_rng = get_truncnorm(-5.0, 5.0, 0.0, 1)

    img_augs = []
    for i in range(n_sample):
        img_aug = np.copy(img)
        angle = angle_rng.rvs()
        scale = scale_rng.rvs(size=2)
        shear = shear_rng.rvs()
        trans = trans_rng.rvs(size=2)
        img_aug = skimage.transform.rotate(img_aug, angle=angle, mode="edge", preserve_range=True)
        img_aug = skimage.transform.warp(img_aug,
                                         skimage.transform.AffineTransform(translation=trans,
                                                                           scale=scale,
                                                                           shear=shear),
                                         mode="edge", preserve_range=True)
        img_augs.append(img_aug.astype(np.float32))
    return img_augs


def convert_rgb2xxx(img, cspace):
    if cspace != 'RGB':
        cvt = getattr(cv2, 'COLOR_RGB2{}'.format(cspace))
        img = cv2.cvtColor(img, cvt)
    return np.copy(img)

# Define a function to return HOG features and visualization


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis is True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_image_features(img, cspace='RGB', spatial_size=(32, 32),
                           hist_bins=32, orient=9,
                           pix_per_cell=8, cell_per_block=2, hog_channel=0,
                           spatial_feat=True, hist_feat=True, hog_feat=True):

    image_features = []

    # apply color conversion if other than 'RGB'
    feature_image = convert_rgb2xxx(img, cspace)

    # Compute spatial feature if flag is set
    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Append features to list
        image_features.append(spatial_features)

    # Compute histogram features if flag is set
    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=(0, 1))
        # Append features to list
        image_features.append(hist_features)

    # Compute HOG features if flag is set
    if hog_feat is True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append features to list
        image_features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(image_features)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block,
              hog_channel, spatial_size, hist_bins):

    bboxes = []
    img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_rgb2xxx(img_tosearch, cspace=cspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = (hog_feat1, hog_feat2, hog_feat3)
            if hog_channel == 'ALL':
                hog_features = np.hstack(hog_features)
            else:
                hog_features = hog_features[hog_channel]

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins, bins_range=(0, 1))
            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bboxes.append(((xbox_left, ytop_draw + ystart),
                               (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return bboxes


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    draw_img = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image

        cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return draw_img
