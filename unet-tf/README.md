It seems the code is copied from https://github.com/zhixuhao/unet

# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

The original dataset is from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/), and I've downloaded it and done the pre-processing.

You can find it in folder data/membrane.

### Data augmentation

The data for training contains 30 512*512 images, which are far not enough to feed a deep learning neural network. I use a module called ImageDataGenerator in keras.preprocessing.image to do data augmentation.

See dataPrepare.ipynb and data.py for detail.


### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy.


---

## How to use

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0

Also, this code should be compatible with Python versions 2.7-3.5.

### Run main.py

You will see the predicted results of test image in data/membrane/test

### Or follow notebook trainUnet



### Results

Use the trained model to do segmentation on test images, the result is statisfactory.

![img/0test.png](img/0test.png)

![img/0label.png](img/0label.png)


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.

## Install Keras/Tensorflow with GPU support on Ubuntu 18.04 with NVIDIA GPU. Refer to https://www.tensorflow.org/install/gpu

### Add NVIDIA package repositories
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
```

### Install NVIDIA driver
```
sudo apt-get install --no-install-recommends nvidia-driver-450
```
### Reboot. Check that GPUs are visible using the command: nvidia-smi

### Install development and runtime libraries (~4GB)
```
sudo apt-get install --no-install-recommends     cuda-10-1     libcudnn7=7.6.5.32-1+cuda10.1    libcudnn7-dev=7.6.5.32-1+cuda10.1
```

### Install TensorRT. Requires that libcudnn7 is installed above.
```
sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 libnvinfer-dev=6.0.1-1+cuda10.1 libnvinfer-plugin6=6.0.1-1+cuda10.1
```

### Install Python3 and virtual env

```
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```

### Install Tensorflow, Keras and required scikit-image. Refer to https://www.tensorflow.org/install/pip (only need to do once). 
```
python3 -m venv --system-site-packages ./tensorflow-venv #create virtual environment
source ./tensorflow-venv/bin/activate
pip install --upgrade pip
pip install scikit-image keras tensorflow

deactivate # exit the virtual env

```

### Run the u-net training model using the previously created virtual env
```
source ./tensorflow-venv/bin/activate
python main.py
deactivate # exit the virtual env
```

