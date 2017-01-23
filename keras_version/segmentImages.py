'''
Segment images with trained CNNs
'''

import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES
import os
import numpy as np

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, dice_jaccard_indices
from model_zoo import sparse_bn_feature_net_81x81 as cyto_fn

# specify directories
direc_name = '/media/jkimmel/HDD0/deepstain/musc_tdtomato/'
data_location = os.path.join(direc_name, 'dic', 'val') # raw images
mask_location = os.path.join(direc_name, 'mask', 'val') # masks
seg_location = os.path.join(direc_name, 'seged') # segmented outputs

# specify channel and feature names
channel_names = ['DIC']
feature_names = ['segMask']

# specify locations and prefixes of trained network weights
trained_network_dir = '/home/jkimmel/src/DeepCell/trained_networks/'
cyto_prefix = '20170123_musc_81x81_bn_feature_net_81x81_' # prefix of network weight filenames
nb_networks = 3 # number of trained networks to ensemble

# set window sizes
window_x, window_y = 40, 40
image_size_x, image_size_y = get_image_sizes(data_location, channel_names[0])

# create a list of weights from all trained networks
list_of_cyto_weights = []
for j in range(nb_networks):
    cyto_weights = os.path.koin(trained_network_dir, cyto_prefix + str(j) + ".h5")
    list_of_cyto_weights.append(cyto_weights)

# run models
predictions = run_models_on_directory(
        data_location=data_location,
        channel_names=channel_names,
        output_location=seg_location,
        model_fn = cyto_fn,
        list_of_weights = list_of_cyto_weights,
        image_size_x = image_size_x,
        image_size_y = image_size_y,
        win_x = window_x,
        win_y = window_y,
        split = False)
