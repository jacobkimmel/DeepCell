'''
Functions to enable training of Keras models
in DeepCell-like format
'''

from customIterators import *

import datetime
import os
import numpy as np

def lr_schedule(rate=0.01, decay = 0.95):
    '''
    Generates a schedule function with simple exp decay,
    takes an integer epoch and outputs a learning rate.

    LearnRate(epoch) = InitRate * epoch**DecayCoeff

    Parameters
    ----------
    rate : float.
        learning rate.
    decay : float.
        decay coefficient
    '''
    def sched(epoch):
        e = np.int32(epoch)
        new_rate = rate * epoch**decay
        return new_rate
    return sched

def load_training_data(file_name, p_val=0.1):
    '''
    Loads training data from a compressed NPZ and returns

    Parameters
    ----------
    file_name : string.
        path to training data NPZ file with the following arrays
            channels, batch, y, pixels_x, pixels_y, win_x, win_y
    p_val : float, optional.
        proportion of data to be used for testing.

    Returns
    -------
    train_dict : dictionary.
        keyed by channels, batch, pixels_x, pixels_y, labels, win_x, win_y
        used with custom ImageDataGenerator to generate minibatches
    '''

    # Load training data as seperate arrays
    training_data = np.load(file_name)
    channels = training_data["channels"]
    batch = training_data["batch"]
    labels = training_data["y"]
    pixels_x = training_data["pixels_x"]
    pixels_y = training_data["pixels_y"]
    win_x = training_data["win_x"]
    win_y = training_data["win_y"]

    total_batch_size = len(labels)
    num_test = np.int32(np.floor(total_batch_size * p_val))
    num_train = np.int32(total_batch_size - num_test)
    # full batch size may differ from total, due to rounding in np.floor()
    full_batch_size = np.int32(num_test + num_train)

    # Split training data and validation data

    arr = np.arange(len(labels))
    arr_shuff = np.random.permutation(arr)

    train_ind = arr_shuff[0:num_train]
    test_ind = arr_shuff[num_train:num_train+num_test]

    X_test, y_test = data_generator(channels.astype("float32"), batch[test_ind], pixels_x[test_ind], pixels_y[test_ind], labels[test_ind], win_x = win_x, win_y = win_y)
    train_dict = {"channels": channels.astype("float32"), "batch": batch[train_ind], "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], "labels": labels[train_ind], "win_x": win_x, "win_y": win_y}

    return train_dict, (X_test, y_test)


def train_model(model=None, dataset=None, exp_name=None, nb_epoch=50,
                callbacks=None, p_val=0.1,
                dir_save='${HOME}', dir_load='${HOME}',
                flip=True, rotate=True, shear=0, class_weight=None):
    '''
    Trains a provided Keras model on a dataset in DeepCell-like format

    Parameters
    ----------
    model : keras model object.
        model architecture to be trained.
    dataset : string.
        name of the dataset used for training.
    exp_name : string, optional.
        name of the training exp.
    nb_epoch : integer, optional.
        number of epochs to perform during training.
    callbacks : list of objects, optional.
        list of callback objects passed to the model.fit() method.
        Default = EarlyStopping, ModelCheckpoint, LearningRateScheduler
    p_val : float, optional.
        proportion of data to be used for testing.
    dir_save : string, optional.
        directory to save model outputs.
        Default = $HOME
    dir_load : string, optional.
        directory to load model outputs.
        Default = $HOME
    flip : boolean, optional.
        flip images during training.
    rotate : boolean, optional.
        rotate images during training.
    shear : float [0, pi], optional.
        randomly shear images in the range [-shear, shear] radians
        during training.
        set == 0 for no shearing.
    class_weight : dictionary, optional.
        keyed by class indices (integers), valued by weights (floats).
        Used to bias the loss function toward certain class.
        Useful for cases where certain classes are underrepresented.

    Returns
    -------
    None
    '''

    # Set output names
    todays_date = datetime.datetime.now().strftime("%Y%m%d")
    file_save = os.path.join(dir_save, todays_date + exp_name + '.h5')
    file_save_loss = os.path.join(dir_save, todays_date + exp_name + '_loss.h5')

    # Load training data
    train_dict, (X_test, Y_test) = load_training_data(training_data_file_name, p_val=p_val)

    # Print training information
    print('Channel image shape:', train_dict["channels"].shape)
    print(train_dict["pixels_x"].shape[0], 'training samples')
    print(X_test.shape[0], 'testing samples')

    # determine the number of classes
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    # convert class vectors to binary class matrices
    train_dict["labels"] = np_utils.to_categorical(train_dict["labels"], n_classes)
    Y_test = np_utils.to_categorical(Y_test, n_classes)

    # set callbacks if none are given
    if callbacks == None:
        callbacks = [EarlyStopping(monitor='val_acc', patience=3),
                    LearningRateScheduler(lr_schedule(rate=0.01, decay=0.95)),
                    ModelCheckpoint(file_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')]

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Instantiate ImageDataGenerator
    # n.b. using custom ImageDataGenerator
    datagen = ImageDataGenerator(
        rotate = rotate,  # randomly rotate images by 90 degrees
        shear_range = shear, # randomly shear images in the range (radians , -shear_range to shear_range)
        horizontal_flip = flip,  # randomly flip images
        vertical_flip = flip)  # randomly flip images

    # fit the model
    loss_history = model.fit_generator(
                        datagen.sample_flow(train_dict, batch_size=batch_size),
                        samples_per_epoch=len(train_dict["labels"]),
                        nb_epoch=n_epoch,
                        validation_data=(X_test, Y_test),
                        class_weight = class_weight,
                        callbacks = callbacks)

    np.savez(file_name_save_loss, loss_history = loss_history.history)
