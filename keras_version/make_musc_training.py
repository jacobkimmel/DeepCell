'''
Make MuSC training data
'''
from __future__ import print_function
from trainingData import *
direc_name = '/netapp/home/jkimmel/data/musc_tdtomato'
feature_names = ['segMask']
channel_names = ['DIC']
window_x = 40
window_y = 40
max_direcs = 50
max_training_examples = 10**6
file_name_save = '/netapp/home/jkimmel/tensorflow/DeepCell/trained_networks/musc_set1.npz'

print("Loading channel data...")
channels = load_channel_imgs(direc_name, channel_names, window_x, window_y, max_direcs)
print("Loading feature masks...")
feature_mask = load_feature_masks(direc_name, feature_names, window_x, window_y, max_direcs)
print("Identifying training pixels...")
min_num = determine_min_examples(feature_mask, window_x, window_y)

feature_matrix = identify_training_pixels(feature_mask, min_num, window_x, window_y, max_training_examples)
print("Saving training data to :" + file_name_save)
save_training_data(file_name_save, channels, feature_matrix, window_x, window_y)
