'''
Segment images with trained CNNs
'''

import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES
import os
import numpy as np
import argparse

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, dice_jaccard_indices
from model_zoo import sparse_bn_feature_net_81x81 as fnet

# set argparsing
parser = argparse.ArgumentParser('Segment images using trained CNNs')
parser.add_argument('data_dir', action='store', default=['./'], help = 'Path to raw images')
parser.add_argument('output_dir', action='store', default=['./'], help = 'Output path for segmented images')
parser.add_argument('--channel_names', action='store', default=['DIC'], help = 'String designating raw image filenames')
parser.add_argument('--net_dir', action='store', default=['./'], help = 'Path to network weights')
parser.add_argument('--net_prefix', action='store', default=['net'], help = 'Prefix of network weights files')
parser.add_argument('--nb_networks', action='store', default=1, help = 'Number of networks to be ensembled')
parser.add_argument('--window_sz', action='store', default=40, help='(Size of receptive fields / 2) - 1')
args = parser.parse_args()
# parse args
data_location = str(args.data_dir)
print("Data location: ", data_location)
seg_location = str(args.output_dir)
print("Output location: ", seg_location)
channel_names = [str(args.channel_names)]
print("Channel names: ", channel_names)
trained_network_dir = str(args.net_dir)
print("Network location: ", trained_network_dir)
net_prefix = str(args.net_prefix)
print("Network weights: ", os.path.join(trained_network_dir, net_prefix + '.h5'))
nb_networks = int(args.nb_networks)
print("Number of networks: ", nb_networks)
window_sz = int(args.window_sz)
print("Window size: ", window_sz)

'''
# specify directories
direc_name = '/media/jkimmel/HDD0/deepstain/musc_tdtomato/'
data_location = os.path.join(direc_name, 'dic', 'val', 'tmp') # raw images
mask_location = os.path.join(direc_name, 'mask', 'val') # masks
seg_location = os.path.join(direc_name, 'seged') # segmented outputs

# specify channel and feature names
channel_names = ['DIC']
feature_names = ['segMask']

# specify locations and prefixes of trained network weights
trained_network_dir = '/home/jkimmel/src/DeepCell/trained_networks/'
net_prefix = '2017-01-23_musc_81x81_100imgs_20170123_musc_81x81_' # prefix of network weight filenames
nb_networks = 1 # number of trained networks to ensemble
'''
# set window sizes
window_x, window_y = window_sz, window_sz
image_size_x, image_size_y = get_image_sizes(data_location, channel_names[0])

# create a list of weights from all trained networks
list_of_cyto_weights = []
for j in range(nb_networks):
    cyto_weights = os.path.join(trained_network_dir, net_prefix + str(j) + ".h5")
    list_of_cyto_weights.append(cyto_weights)

# run models
predictions = run_models_on_directory(
        data_location=data_location,
        channel_names=channel_names,
        output_location=seg_location,
        model_fn = fnet,
        list_of_weights = list_of_cyto_weights,
        n_features = 2,
        image_size_x = image_size_x,
        image_size_y = image_size_y,
        win_x = window_x,
        win_y = window_y,
        split = True)
