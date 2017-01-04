'''
Generate training data for DeepCell-like networks
'''
from __future__ import print_function
import numpy as np
from cnn_functions import format_coord as cf
from skimage import morphology as morph
import skimage as sk
import scipy as sp
from scipy import ndimage
from skimage import feature
from cnn_functions import get_image
import glob
import os
import fnmatch

max_training_examples = 1*10**7
# number of pixels to span from the center pixels in either direction
# window_x, window_y = 50, 50 will yield 101 x 101 px image
window_x = 50
window_y = 50

direc_name = '$HOME/data/musc_tdtomato/training/'
set_name = 'musc_tdtomato_101x101.npz'
file_name_save = os.path.join('$HOME/data/musc_tdtomato/training_npz/', set_name)
channel_names = ["DIC"]

def load_channel_imgs(direc_name, channel_names, window_x = 50, window_y = 50, norm = "median", smooth = "average"):
    '''
    Load channel images from a directory with a standardized naming scheme into
    a 4D ndarray for training

    Performs image normalization on all channel images (mean or median norm.)
    Smooths images with an averaging or Gaussian kernel

    Parameters
    ----------
    direc_name : string.
            name of the directory containing training channel images
            (where channel images are brightfield, fluorescent images).
    channel_names : list of strings.
            list of strings identifying the types of channel images in the
            directory and matching the filenaming scheme
            i.e. ["DIC", "DAPI", "FITC"] or ["phase", "gfp"], etc.
    window_x : integer, optional.
            number of pixels on either side of the interrogated pixel to
            sample for classification in X
            (used as the size of an averaging filter)
    window_y : integer, optional.
            number of pixels on either side of the interrogated pixel to
            sample for classification in Y
            (used as the size of an averaging filter)
    norm : string, optional.
            string specifying normalization metric to use.
            Options : "median", "mean"
            Images are normalized by division of the provided metric.
    smooth : string, optional.
            String specifying smoothing method.
            Options: "average", "gaussian".
            "average" convolves a [2*window_x+1, 2*window_y+1] filter.
            "gaussian" convolves a gaussian filter, sigma = 2.

    Returns
    -------
    channels : ndarray.
            4-dimensional ndarray of images with the following format
            channels.shape = [directory_number, channel_number, img_x, img_y]
    '''

    num_channels = len(channel_names)
    num_direcs = 1 # hardcode to 1 when training from one large directory

    imglist = os.listdir(direc_name)

    # load a temp image to get the image size
    # van valen's get_image() checks file types and uses the tifffile lib if the
    # file is a tiff, otherwise falling back to skimages io.imread()
    img_tmp = get_image( os.path.join( direc_name, imglist[0] ) )
    image_size_x, image_size_y = img_tmp.shape
    # rm temp image from memory
    del img_tmp

    # init an array for the predictive images
    channels = np.zeros((num_direcs, num_channels, image_size_x, image_size_y), dtype='float32')

    # Load channels
    direc_counter = 0 # left in place in case multidir training is added
    channel_counter = 0
    for channel in channel_names:
        for img in imglist:
            # check if filename has the name of the channel in it
            if fnmatch.fnmatch(img, r'*' + channel + r'*'):
                channel_file = img
                channel_file = os.path.join(training_direcs, channel_file)
                channel_img = get_image(channel_file)

                # Normalize the images
                if norm == 'median':
                    norm_coeff = np.percentile(channel_img, 50)
                else:
                    norm_coeff = np.mean(channel_img)
                channel_img /= norm_coeff

                # Convolute with an averaging kernel to smooth noise
                if smooth == 'average':
                    avg_kernel = np.ones((2*window_size_x + 1, 2*window_size_y + 1))
                    channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
                else:
                    channel_img = ndimage.gaussian_filter(channel_img, sigma=2)

                # set channel image as the final two dimensions in a 4D ndarray
                channels[direc_counter,channel_counter,:,:] = channel_img
                channel_counter += 1

    return channels

num_of_features = 1
num_channels = len(channel_names)
num_direcs = 1 # hardcode to 1 when training from one large directory

def load_feature_masks(direc_name, feature_names, window_x = 50, window_y = 50):
    '''
    Parameters
    ----------
    direc_name : string.
            name of the directory containing training channel images
            where channel images are brightfield, fluorescent images.
    feature_names : list of strings.
            list of strings identifying the names of feature masks in the
            directory and matching the filenaming scheme.
            i.e. ["cytoplasm", "nuclei"] or ["dog", "cat"], etc.
    window_x : integer, optional.
            number of pixels on either side of the interrogated pixel to
            sample for classification in X.
    window_y : integer, optional.
            number of pixels on either side of the interrogated pixel to
            sample for classification in Y.

    Return
    ------
    feature_mask : ndarray.
            4-dimensional ndarray of images with the following format
            feature_mask.shape = [directory_number, feature_number, mask_x, mask_y]

            where the final feature_number is the background class.

    '''

    imglist = os.listdir(direc_name)

    # load a temp image to get the image size
    # van valen's get_image() checks file types and uses the tifffile lib if the
    # file is a tiff, otherwise falling back to skimages io.imread()
    img_tmp = get_image( os.path.join( direc_name, imglist[0] ) )
    image_size_x, image_size_y = img_tmp.shape
    # rm temp image from memory
    del img_tmp

    num_direcs = 1 # hardcode
    direc_counter = 0 # left in place to add multidir training later
    num_of_features = len(feature_names)
    # init array for the feature masks
    feature_mask = np.zeros((num_direcs, num_of_features + 1, image_size_x, image_size_y))

    # Load feature mask
    for j in range(num_of_features):
        feature_name = feature_names[j]
        for img in imglist:
            if fnmatch.fnmatch(img, feature_name):
                feature_file = os.path.join(direc_name, direc, img)
                feature_img = get_image(feature_file)

                if np.sum(feature_img) > 0:
                    feature_img /= np.amax(feature_img)

                feature_mask[direc_counter,j,:,:] = feature_img

    # Compute the mask for the background
    feature_mask_sum = np.sum(feature_mask[direc_counter,:,:,:], axis = 0)
    feature_mask[direc_counter,num_of_features,:,:] = 1 - feature_mask_sum

    return feature_mask

def determine_min_examples(feature_mask, window_x, window_y):
    '''
    Finds the feature with the minimum number of pixels to be classified for
    use in class balancing

    Parameters
    ----------
    feature_mask : ndarray.
            4-dimensional ndarray of images with the following format
            feature_mask.shape = [directory_number, feature_number, mask_x, mask_y]
            where the final feature_number is the background class.
    window_x : integer, optional.
            number of pixels on either side of the interrogated pixel to
            sample for classification in X.
    window_y : integer, optional.
            number of pixels on either side of the interrogated pixel to
            sample for classification in Y.

    Returns
    -------
    min_num : integer.
        minimum number of pixels in a feature.
    '''
    # trim 1/2 filter width from the edge of each feature map to ignore these
    # regions during segmentation
    feature_mask_trimmed = feature_mask[:,:,window_x+1:-window_x-1,window_y+1:-window_y-1]
    px_counts = np.zeros([feature_mask.shape[1], 1])
    for i in range(feature_mask.shape[0]):
        for j in range(feature_mask.shape[1]-1):
            px_counts[j] = np.sum(feature_mask_trimmed[i,j,:,:])

    min_num = np.min(px_counts)
    return min_num

def identify_training_pixels(channels, feature_mask, min_num, max_training_examples = np.Inf):
    '''
    Identifies pixels to use for training using randomization & class balancing

    Parameters
    ----------
    channels : ndarray.
        4-dimensional ndarray of images with the following format
        channels.shape = [directory_number, channel_number, img_x, img_y]
    feature_mask : ndarray.
        4-dimensional ndarray of images with the following format
        feature_mask.shape = [directory_number, feature_number, mask_x, mask_y]
        where the final feature_number is the background class.
    min_num : integer.
        minimum number of pixels in a feature, i.e. the smallest class.
        used to class balance the training pixels.
    max_training_examples : integer, optional.
        maximum number of examples to use. Default = Inf.

    Returns
    -------
    feature_rows : list of integers.
        row indices for pixels to train on.
    feature_cols : list of integers.
        col indices for pixels to train on.
    feature_batch : list of integers.
        batch indices for pixels to train on. Only neccessary if subdirectories
        in the training directory are utilized.
    feature_label : list of integers.
        ground truth class labels for pixels to train on.
    '''


    # init lists storing row, col locations, batch, and ground truth labels
    feature_rows = []
    feature_cols = []
    feature_batch = []
    feature_label = []

    for direc in xrange(channels.shape[0]):

        for k in xrange(feature_mask.shape[1]):
            feature_counter = 0
            # identify row and col locations of True pixels in all feature masks
            feature_rows_temp, feature_cols_temp = np.where(feature_mask[direc,k,:,:] == 1)

            # Check to make sure the features are actually present
            if len(feature_rows_temp) <= 0:
                print("feature_rows_temp was empty, exiting")
                return

            # Randomly permute index vector
            # get an array of indices in order
            non_rand_ind = np.arange(len(feature_rows_temp))
            # randomly shuffle this ordered array of indices
            rand_ind = np.random.choice(non_rand_ind, size = len(non_rand_ind), replace = False)

            for i in rand_ind:
                if feature_counter < min_pixel_counter:
                    if (feature_rows_temp[i] - window_size_x > 0) and (feature_rows_temp[i] + window_size_x < image_size_x):
                        if (feature_cols_temp[i] - window_size_y > 0) and (feature_cols_temp[i] + window_size_y < image_size_y):
                            feature_rows += [feature_rows_temp[i]]
                            feature_cols += [feature_cols_temp[i]]
                            feature_batch += [direc]
                            feature_label += [k]
                            feature_counter += 1

    feature_rows = np.array(feature_rows, dtype = 'int32')
    feature_cols = np.array(feature_cols, dtype = 'int32')
    feature_batch = np.array(feature_batch, dtype = 'int32')
    feature_label = np.array(feature_label, dtype = 'int32')

    # Randomly select training points if there are more than a specified
    # maximum number of examples
    if len(feature_rows) > max_training_examples:
    	non_rand_ind = np.arange(len(feature_rows))
    	rand_ind = np.random.choice(non_rand_ind, size = max_training_examples, replace = False)

    	feature_rows = feature_rows[rand_ind]
    	feature_cols = feature_cols[rand_ind]
    	feature_batch = feature_batch[rand_ind]
    	feature_label = feature_label[rand_ind]

    return feature_rows, feature_cols, feature_batch, feature_label

def save_training_data(file_name_save, channels, feature_rows, feature_cols, feature_batch, feature_label, window_x=50, window_y=50):
    '''
    Saves training data in an NPZ structure.

    Parameters
    ----------
    file_name_save : string.
        filename to save.
    channels : ndarray.
        4-dimensional ndarray of images with the following format
        channels.shape = [directory_number, channel_number, img_x, img_y]
    feature_rows : list of integers.
        row indices for pixels to train on.
    feature_cols : list of integers.
        col indices for pixels to train on.
    feature_batch : list of integers.
        batch indices for pixels to train on. Only neccessary if subdirectories
        in the training directory are utilized.
    feature_label : list of integers.
        ground truth class labels for pixels to train on.
    window_x : integer, optional.
            number of pixels on either side of the interrogated pixel to
            sample for classification in X.
    window_y : integer, optional.
            number of pixels on either side of the interrogated pixel to
            sample for classification in Y.

    Returns
    -------
    None
    '''


    np.savez(file_name_save, channels = channels, y = feature_label,
            batch = feature_batch, pixels_x = feature_rows,
            pixels_y = feature_cols, win_x = window_x, win_y = window_y)

    return
