'''
Train CNNs for semantic segmentation of MEF nuclei
'''

from __future__ import print_function
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping

from cnn_functions import rate_scheduler, train_model_sample
from model_zoo import bn_feature_net_61x61 as the_model

import os
import datetime
import numpy as np

# define a batch size and the number of epochs to run
batch_size = 512
n_epoch = 30

# specify names of training set and directory to save model
dataset = "mef_150imgs"
expt = "20170131_mef_61x61"

direc_save = "/home/jkimmel/src/DeepCell/trained_networks/"
direc_data = "/media/jkimmel/HDD0/deepstain/mef_nuclei/20161212_DAPI_MycRas/20161212_200506_893/"

# Set the optimizer
# SGD works best for batchnorm nets, while RMSprop seems to be better
# for non-normalized nets

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.95)


for iterate in range(0,2):

    model = the_model(n_channels = 1, n_features = 2, reg = 1e-5)

    train_model_sample(model = model, dataset = dataset, optimizer = optimizer,
        expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
        direc_save = direc_save,
        direc_data = direc_data,
        lr_sched = lr_sched,
        rotate = True, flip = True, shear = False, early_stopping=False)

    del model
    # reset the keras numbering scheme to ensure layers are named properly
    # when training >1 model in a run
    from keras.backend.common import _UID_PREFIXES
    for key in _UID_PREFIXES.keys():
        _UID_PREFIXES[key] = 0
