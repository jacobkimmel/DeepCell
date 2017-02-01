# DeepCell

@jacobkimmel's modification of @vanvalen's DeepCell networks and associated tooling.

# Usage

## Training Data Preparation

Training data is preprocessed and saved as a compressed `numpy` array prior to training. Preprocessing consists of image normalization and denoising by a rolling averaging filter the same size as the network's receptive field.

Raw images and binary ground truth segmentation masks can be saved in the same directory with identifying strings in their filenames, or in separate directories.

i.e.:
```
img1_raw.tif
img1_seg.tif
img2_raw.tif
img2_seg.tif
...
```

See `keras_version/make_mef_training.py` for an example script using the training data generation functions.

Training data preparation functions allow for parallel processing of training images across multiple cores, unlike the original scripts.

## Training

Networks designed to take receptive field-sized inputs are trained using @vanvalen's custom ImageDataGenerator that allows for image patch generation from a set of source images and training pixel coordinates.

See `keras_version\mef_train.py` for an example of a training script.

Model weights are saved to a specified directory in HDF5 format (`.h5`), and loss history is saved as a compressed `numpy` array.

## Prediction

Prediction networks designed to take raw image sized inputs and incorporating atrous kernels (*d*-regularly sparsed kernels, dilated kernels) are able to use the weights of receptive-field sized models trained on patches.

To predict the feature map of new images, use the `keras_version/segmentImages.py` command line tool.
