# unet-caffe
Thus is unet for the [ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/)

## UNet using Caffe

### Clone Repo: https://github.com/passlab/unet-caffe

### Prepare images for training (http://brainiac2.mit.edu/isbi_challenge/). 
1. Download images from http://brainiac2.mit.edu/isbi_challenge/
1. Get the image out of the tif file and rename the training, test and labeled image
1. Generate more data according to the Machine Learning method (data augmentation method: such as scale,flip, rotation, and so on). 

You can perform Augmentation using Keras


1. You need to install keras and tensor flow to be able to perform deep learning. You can use pip to install these 2 libraries
2. Refer following to find the code to perform Augmentation https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/


### Install Caffe on /opt/mlframework/cafee (Siddhesh does not to do this, but keep your instruction here as well). 
1. Pre-installed environment by command: 

```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
```
1. Install CUDA(https://developer.nvidia.com/cuda-downloads) and cuDNN(https://developer.nvidia.com/cudnn) from Nvidia 

1. Install python and python environments

```
for req in $(cat requirements.txt); do pip install $req; done

```
1. Compilation with Make

```
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)
make all
make test
make runtest

```

1. Install pycaffe

```
make pycaffe

```

### Creating UNet model based on ... (Siddhesh does not need to do this one, but keep your instruction here as well)

1. Prepar your own data

1. Create your own model based on your data structure and save your model structure in traning.prototxt. (https://caffe.berkeleyvision.org/tutorial/layers.html)

1. Create your training configure as solver.prototxt

1. Train you network

### Use the UNet network to train the model and generate the model files that are needed by Xilinx Vitis-AI

1. cd into the unet folder

1. change the data path to your data stored location in imglist.txt

1. change your traning.prototxt path and model weight save path in the solver.prototxt

1. change the Caffe path and python path in the train.sh

1. ./train.sh

1. start your training

### Use Caffe UNet to do segmentation using GPU and CPU on Carina
