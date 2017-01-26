'''
Make MuSC training data
'''
from __future__ import print_function
from trainingData import *
import os
direc_name = '/media/jkimmel/HDD0/deepstain/mef_nuclei/20161212_DAPI_MycRas/20161212_200506_893'
dic_dir = os.path.join(direc_name, 'dic')
mask_dir = os.path.join(direc_name, 'masks')
feature_names = ['segMask']
channel_names = ['DIC']
window_x = 40
window_y = 40
max_direcs = 150
max_training_examples = 2*10**6
file_name_save = os.path.join(direc_name, 'mef_150imgs.npz')

print("Loading channel data...")
channels = load_channel_imgs(dic_dir, channel_names, window_x, window_y, max_direcs)
print("Loading feature masks...")
feature_mask = load_feature_masks(mask_dir, feature_names, window_x, window_y, max_direcs)
print("Identifying training pixels...")
min_num = determine_min_examples(feature_mask, window_x, window_y)

rows, cols, batch, label = identify_training_pixels(feature_mask, min_num, window_x, window_y, max_training_examples)
print("Saving training data to :" + file_name_save)
save_training_data(file_name_save, channels, rows, cols, batch, label, window_x, window_y)
