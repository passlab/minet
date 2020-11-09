from __future__ import print_function

import os, sys, argparse
import subprocess
import tensorflow as tf
import numpy as np
import cv2

# Bring in VAI Quantizer, Compiler, and Partitioner
from vai.dpuv1.rt.xdnn_rt_tf import TFxdnnRT as xdnnRT
from vai.dpuv1.rt.xdnn_util import dict2attr, list_depth, make_list
from utils import get_input_fn, top5_accuracy, LABEL_OFFSET, BATCH_SIZE
from shutil import rmtree


XCLBIN = '/opt/xilinx/overlaybins/xdnnv3/'
if (not os.path.exists(XCLBIN) and 'VAI_ALVEO_ROOT' in os.environ and
        os.path.isdir(os.path.join(os.environ['VAI_ALVEO_ROOT'], 'overlaybins/xdnnv3'))):
    # Environment Variables (obtained by running "source overlaybins/setup.sh")
    XCLBIN = os.path.join(os.environ['VAI_ALVEO_ROOT'], 'overlaybins/xdnnv3')

def get_default_compiler_args():
    return {
        'dsp':                  96,
        'memory':               9,
        'bytesperpixels':       1,
        'ddr':                  256,
        'data_format':          'NHWC',
        'mixmemorystrategy':    True,
        'pipelineconvmaxpool':  True,
        'xdnnv3':               True,
        'usedeephi':            True,
        'pingpongsplit':        True,
        'deconvinfpga':         True,
        'quantz':               '',
        'fcinfpga':             True,
        'bridges': ['bytype', 'Concat'],
    }

rt = xdnnRT(None,
                networkfile='/workspace/tutorial/Vitis-AI-Tutorials/files/workspace/tf_chkpts/unet1/frozen_graph.pb',
                quant_cfgfile='/workspace/tutorial/Vitis-AI-Tutorials/files/workspace/tf_chkpts/unet1/unet1_fix.txt',
                batch_sz='10',
                startnode='input_1',
                finalnode='onv2d_19/Relu',
                xclbin=XCLBIN,
                device='FPGA',
                placeholdershape="{'input_1' : [1,224,224,3]}",
                savePyfunc=True,
                **get_default_compiler_args()
               )
