'''
Plot keras model training data
'''

import numpy as np
import matplotlib.pyplot as plt

def get_loss_dict(loss_file):
    ll = np.load(loss_file)
    l = ll['loss_history']
    loss = l.any()

    return loss

def plot_accuracy(loss, save_name=None, exp_name=None):
    plt.plot(loss['acc'])
    plt.plot(loss['val_acc'])
    plt.title('Model Accuracy')
    if exp_name:
        plt.suptitle(exp_name)
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Training Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    if save_name:
        plt.savefig(save_name)

def plot_loss(loss, save_name=None, exp_name=None):
    plt.plot(loss['loss'])
    plt.plot(loss['val_loss'])
    plt.title('Model Loss')
    if exp_name:
        plt.suptitle(exp_name)
    plt.ylabel('Loss')
    plt.xlabel('Training Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    if save_name:
        plt.savefig(save_name)
