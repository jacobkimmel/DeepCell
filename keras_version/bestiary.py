'''
Atrous kernel models for semantic segmentation
customized by: @jacobkimmel
credit to: @vanvalen
'''
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Activation, BatchNormalization, Flatten
from keras.layers import Input, Dense, Lambda, Merge
from cnn_functions import load_training_data, euclidean_distance, eucl_dist_output_shape, tensorprod_softmax, sparse_Convolution2D, sparse_MaxPooling2D, TensorProd2D, set_weights, residual_block
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop

def bn_feature_net_61x61(n_features = 2, n_channels = 1, reg = 1e-5, init = 'he_normal'):
	print("Using feature net 61x61 with batch normalization")
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=(n_channels, 61, 61), W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(200,4,4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

def sparse_bn_feature_net_61x61(batch_input_shape = (1,2,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()
	d = 1
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d)))
	d *= 2

	model.add(sparse_Convolution2D(200, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model

def bn_feature_net_81x81(n_features = 2, n_channels = 1, reg = 1e-5, init = 'he_normal'):
	print("Using feature net 81x81 with batch normalization")
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=(n_channels, 81, 81), W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Convolution2D(200,4,4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

def sparse_bn_feature_net_81x81(batch_input_shape = (1,1,2048,2048), n_features = 2, reg = 1e-5, init = 'he_normal', weights_path = None):

	model = Sequential()

	d = 1
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, batch_input_shape = batch_input_shape, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))) # 1024

	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))) # 512

	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(sparse_MaxPooling2D(pool_size=(2, 2), strides = (d,d))) # 256

	d *= 2
	model.add(sparse_Convolution2D(64, 3, 3, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(sparse_Convolution2D(200, 4, 4, d = d, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, init = init, W_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation(tensorprod_softmax))

	model = set_weights(model, weights_path)

	return model
