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
feature_names = ["segMask"]

def gauss_filter_2D(shape=(8,8),sigma=2):
    """
    2D gaussian mask

    Parameters
    ----------
    shape : tuple of integers.
        height and width of the desired filter.
        for image smoothing, a shape of at least (4*sigma, 4*sigma)
        is recommended.
    sigma : float.
        sigma for the desired gaussian distribution.

    Returns
    -------
    h : ndarray.
        2D ndarray of floats representing the generated Gaussian filter.
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def load_channel_imgs(direc_name, channel_names, window_x = 50, window_y = 50, max_direcs = 100, norm = "median", smooth = "average"):
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
    max_direcs : integer, optional.
            maximum number of unique fields of view to load for training,
            primarily limited to conserve memory.
            Default = 100.
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

    Notes
    -----
    Ensure that max_direcs is set to an appropriate size, or a MemoryError will
    occur when trying to generate a large ndarray to store all the channel images.
    For reference, an empty ndarray of zeros sized for 100 images at 4 megapixels
    ( created with np.zeros(100,1,2024,2024) ) requires 3GB of available memory.

    '''

    num_channels = len(channel_names)
    imglist = glob.glob( os.path.join(direc_name, '*' + channel_names[0] + '*') )

    # count how many occurences of a single channel name appear to find the
    # number of unique images ("direcs" in van valen's parlance)
    num_direcs = len(imglist)

    # load a temp image to get the image size
    # van valen's get_image() checks file types and uses the tifffile lib if the
    # file is a tiff, otherwise falling back to skimages io.imread()
    img_tmp = get_image( imglist[0] )
    image_size_x, image_size_y = img_tmp.shape
    # rm temp image from memory
    del img_tmp

    # set the number of direcs to be loaded as the provided max_direcs if the number
    # present is greater
    if num_direcs > max_direcs:
        load_direcs = max_direcs
    else:
        load_direcs = num_direcs

    # init an array for the predictive images
    channels = np.zeros((load_direcs, num_channels, image_size_x, image_size_y), dtype='float32')

    # Load channels
    channel_counter = 0
    for channel in channel_names:
        imglist_channel = glob.glob(os.path.join(direc_name, '*' + channel + '*'))
        assert len(imglist_channel) == num_direcs # ensure all channels have = img #
        # select only the first N unique fields of view, where N = load_direcs
        imglist_channel = imglist_channel[:load_direcs]
        # direc_counter is actually a 'unique FOV counter'
        # but van valen's parlance is maintained
        direc_counter = 0
        for img in imglist_channel:
            # check if filename has the name of the channel in it
            channel_file = img # glob returns full file paths
            channel_img = get_image(channel_file)

            # Normalize the images
            if norm == 'median':
                norm_coeff = np.percentile(channel_img, 50)
            else:
                norm_coeff = np.mean(channel_img)
            channel_img /= norm_coeff

            # Convolute with a smoothing kernel and subtract from the image
            # to remove local noise at window scale
            if smooth == 'average':
                avg_kernel = np.ones((2*window_x + 1, 2*window_y + 1))
                channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
            else:
                sigma = 2
                gauss_kernal = gauss_filter_2D((2*window_x + 1, 2*window_y + 1), sigma = sigma)
                channel_img -= ndimage.convolve(channel_img, gauss_kernel)


            # set channel image as the final two dimensions in a 4D ndarray
            channels[direc_counter,channel_counter,:,:] = channel_img
            direc_counter += 1
            print(direc_counter)
        channel_counter += 1

    return channels

def load_feature_masks(direc_name, feature_names, window_x = 50, window_y = 50, max_direcs = 100):
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
    max_direcs : integer, optional.
            maximum number of unique images to load in the training set,
            primarily limited to conserve memory.
            Default = 100.

    Return
    ------
    feature_mask : ndarray.
            4-dimensional ndarray of images with the following format
            feature_mask.shape = [directory_number, feature_number, mask_x, mask_y]

            where the final feature_number is the background class.
    Notes
    -----
    Ensure that max_direcs is set to an appropriate size, or a MemoryError will
    occur when trying to generate a large ndarray to store all the channel image
    For reference, an empty ndarray of zeros sized for 100 images at 4 megapixels
    ( created with np.zeros(100,1,2024,2024) ) requires 3GB of available memory.

    '''

    imglist = glob.glob( os.path.join(direc_name, '*' + feature_names[0] + '*') )

    # load a temp image to get the image size
    # van valen's get_image() checks file types and uses the tifffile lib if the
    # file is a tiff, otherwise falling back to skimages io.imread()
    img_tmp = get_image( imglist[0] )
    image_size_x, image_size_y = img_tmp.shape
    # rm temp image from memory
    del img_tmp

    # count number of unique fields of view from a single feature
    num_direcs = len(imglist)

    # set the number of direcs to be loaded as the provided max_direcs if the number
    # present is greater
    if num_direcs > max_direcs:
        load_direcs = max_direcs
    else:
        load_direcs = num_direcs

    direc_counter = 0 # left in place to add multidir training later
    num_of_features = len(feature_names)
    # init array for the feature masks
    feature_mask = np.zeros((load_direcs, num_of_features + 1, image_size_x, image_size_y))

    # Load feature mask
    for j in range(num_of_features):
        feature_name = feature_names[j]
        direc_counter = 0
        imglist_feature = glob.glob( os.path.join(direc_name, '*' + feature_name + '*') )
        assert len(imglist_feature) == num_direcs # check mask #'s are =
        # truncate imglist_feature to only the first N unique fields of view
        imglist_feature = imglist_feature[:load_direcs]
        for img in imglist_feature:
            feature_file = img # glob.glob returns whole file paths
            # set the mask files to be inverted
            # weird color map bug in skimage / tifffile lib leads to inversion
            # of binary tiff's exported from matlab
            # https://github.com/scikit-image/scikit-image/issues/1940
            feature_img = get_image(feature_file, invert = True)

            if np.sum(feature_img) > 0:
                feature_img /= np.amax(feature_img)

            feature_mask[direc_counter,j,:,:] = feature_img
            direc_counter += 1

    # Compute the mask for the background
    for idx in range(feature_mask.shape[0]):
        feature_mask_sum = np.sum(feature_mask[idx,:,:,:], axis = 0)
        feature_mask[idx,num_of_features,:,:] = 1 - feature_mask_sum

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

    image_size_x, image_size_y = channels.shape[2], channels.shape[3]

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
                if feature_counter < min_num:
                    if (feature_rows_temp[i] - window_x > 0) and (feature_rows_temp[i] + window_x < image_size_x):
                        if (feature_cols_temp[i] - window_y > 0) and (feature_cols_temp[i] + window_y < image_size_y):
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
