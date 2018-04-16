
# UDACITY MLND Capstone Project (Tanbir Ahmed)

## On Classification of the STL-10 dataset using Deep CNN and Transfer Learning

# Content

* ****
1. **Modules to import**
* **Device check**
* **Configuration**
* **Load STL-10 Labeled data**
* **Exploratory visualization of the dataset**
* **Categorize labels**
* **Normalize image data**
* **General Model from scratch**
* **Feature extraction and finetuning using pre-trained Imagenet model (ResNet50)**
* **Augmentation for the Model from Scratch**
* **Performance summary of the CNN model from scratch**
* **Conclusion**
* ****

# Required resources

* **Disk space:** Approximately 8GB (with models and data saved)
* **Memory:** Approximately 16GB 
* **CPU/GPU:** A GPU is recommended for faster training (we used GTX 1060)
* **Libraries:** Keras, Tensorflow, Scikit-learn, OpenCV
* **Data:** ./stl10/train_X.bin, ./stl10/train_y.bin, ./stl10/test_X.bin, ./stl10/test_y.bin, ./stl10/unlabeled_X.bin
    * Data can be downloaded from https://cs.stanford.edu/~acoates/stl10/
* **Directories:**
    * Bottleneck features (for original train data and test data with ResNet50): ./bottleneck_features
    * Augmented train data (for saving augmented training data): ./augmented_train_data
    * Finetuned model saved (for saving finetuned ResNet50 model): ./best_model_saved   

# 1. Modules to import


```python
import numpy as np
from tqdm import tqdm
import os
import shutil
from shutil import copyfile
import scipy
from scipy import misc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

import csv
import random
import glob
import cv2

import keras
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint   
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D, Flatten, BatchNormalization
```

# 2. Device check

We use GeForce GTX 1060 GPU for execution of the code. The following code checks availability of available devices.


```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

    [name: "/cpu:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 4575689030084036791
    , name: "/gpu:0"
    device_type: "GPU"
    memory_limit: 104831385
    locality {
      bus_id: 1
    }
    incarnation: 9037885439629910453
    physical_device_desc: "device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0"
    ]
    

# 3. Configuration


```python
config = {
    'data_directory' : './stl10',
    'augmented_train_data' : './augmented_train_data',
    'ResNet50' : {
        'name' : 'ResNet50',
        'input_shape' : (197, 197, 3),
        'dataset' : 'STL-10',
        'features_file_train' : './bottleneck_features/ResNet50_STL-10_features_train.npz',
        'features_file_test' : './bottleneck_features/ResNet50_STL-10_features_test.npz',
        'model_file_saved' : './best_model_saved/ResNet50_STL-10_model_best.hdf5'
    }
}
```
