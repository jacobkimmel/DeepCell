'''
Make Stentor training data
'''
from __future__ import print_function
from trainingData import *
import os
direc_name = '/media/jkimmel/TM/_ForYou/'
channel_dir = os.path.join(direc_name, 'Images')
mask_dir = os.path.join(direc_name, 'Masks')
feature_names = ['z1_mask']
channel_names = ['z1c0', 'z1c1', 'z1c2']
window_x = 40
window_y = 40
max_direcs = 100
max_training_examples = 2*10**6
file_name_save = os.path.join('/media/jkimmel/HDD0/stentor/', 'stentor_z1.npz')

print("Loading channel data...")
channels = load_channel_imgs(channel_dir, channel_names, window_x, window_y, max_direcs)
print("Loading feature masks...")
feature_mask = load_feature_masks(mask_dir, feature_names, window_x, window_y, max_direcs)
print("Identifying training pixels...")
min_num = determine_min_examples(feature_mask, window_x, window_y)

feature_matrix = identify_training_pixels(feature_mask, min_num, window_x, window_y, max_training_examples)
print("Saving training data to :" + file_name_save)
save_training_data(file_name_save, channels, feature_matrix, window_x, window_y)
