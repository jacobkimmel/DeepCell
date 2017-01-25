'''
DeepCell test training a CNN
'''

from __future__ import print_function
from keras.optimizers import SGD, RMSprop

from cnn_functions import rate_scheduler, train_model_sample
from model_zoo import bn_feature_net_81x81 as the_model

import os
import datetime
import numpy as np

# define a batch size and the number of epochs to run
batch_size = 256
n_epoch = 25

# specify names of training set and directory to save model
dataset = "musc_81x81_100imgs"
expt = "20170123_musc_81x81"

direc_save = "/home/jkimmel/src/DeepCell/trained_networks/"
direc_data = "/media/jkimmel/HDD0/deepstain/musc_tdtomato/"

# Set the optimizer
# SGD works best for batchnorm nets, while RMSprop seems to be better
# for non-normalized nets

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.95)


for iterate in range(1,4):

    model = the_model(n_channels = 1, n_features = 2, reg = 1e-5)

    train_model_sample(model = model, dataset = dataset, optimizer = optimizer,
        expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
        direc_save = direc_save,
        direc_data = direc_data,
        lr_sched = lr_sched,
        rotate = True, flip = True, shear = False)

    del model
    # reset the keras numbering scheme to ensure layers are named properly
    # when training >1 model in a run
    from keras.backend.common import _UID_PREFIXES
    for key in _UID_PREFIXES.keys():
        _UID_PREFIXES[key] = 0
