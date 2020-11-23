# Copyright 2019 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import ctypes
import os
import sys
moddir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.cdll.LoadLibrary('%s/../libdpuv1utils.so' % moddir)

class PyArray(ctypes.Structure):
    _fields_ = [("height", ctypes.c_int),
                ("width", ctypes.c_int),
                ("channels", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_float))]

c_array_p = ctypes.POINTER(PyArray)
c_float_p = ctypes.POINTER(ctypes.c_float)

# Original: tiny_yolo_v3_biases = np.array([10,14,  23,27,  37,58,  81,82,  135,169,  344,319], np.float32)
yolov3_bias_coco = np.array([10,13,16,30,33,23, 30,61,62,45,59,119, 116,90,156,198,373,326], dtype=np.float32)
tiny_yolov3_bias_coco = np.array([23,27,  37,58,  81,82,  81,82, 135,169,  344,319], np.float32)    # for 416x416 input

yolov2_bias_coco = np.array(
    (0.572730004787445068359375, 
     0.677384972572326660156250,
     1.874459981918334960937500,
     2.062530040740966796875000,
     3.338429927825927734375000,
     5.474339962005615234375000,
     7.882820129394531250000000,
     3.527780055999755859375000,
     9.770520210266113281250000,
     9.168279647827148437500000), dtype=np.float32)

yolov2_bias_voc = np.array((1.08,1.19 ,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52), dtype=np.float32)


#  int yolov3_postproc(Array* outputs, int nArrays, float* biases, int net_h, int net_w, int classes, int anchorCnt, 
#                                        int img_height, int img_width, float conf_thresh, float iou_thresh, float** retboxes) {

_lib.yolov3_postproc.restype  = ctypes.c_int
_lib.yolov3_postproc.argtypes = [ctypes.c_void_p, 
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_void_p]

#int yolov2_postproc(Array* output_tensors, int nArrays, const float* biases, int net_h, int net_w, int classes, int anchor_cnt, 
#                                        int img_height, int img_width, float conf_thresh, float iou_thresh, float** retboxes) {

_lib.yolov2_postproc.restype  = ctypes.c_int
_lib.yolov2_postproc.argtypes = [ctypes.c_void_p, 
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_void_p]

# fpga_Output : List of output arrays from FPGA
# config : dict of config params
# img_shape : list of Input image shapes, size = number of images
# biases : numpy 1D array of type np.float32
def yolov3_postproc(fpga_output, config, img_shape, biases=yolov3_bias_coco):
    bbox_list_for_images = []
    nArrays = len(fpga_output)
    batchSz = len(img_shape)
    c_outputs = (PyArray * nArrays)()

    for bsz in range(batchSz):
        for i, arr in enumerate(fpga_output):
            batch = arr[bsz]
            x = PyArray(arr.shape[2], arr.shape[3], arr.shape[1], batch.ctypes.data_as(c_float_p))
            c_outputs[i] = x
        bboxes = c_float_p()

        nboxes = _lib.yolov3_postproc(ctypes.byref(c_outputs),
                            nArrays,
                            biases.ctypes.data_as(c_float_p),
                            config['net_h'],
                            config['net_w'],
                            config['classes'],
                            config['anchorCnt'],
                            img_shape[bsz][0],
                            img_shape[bsz][1],
                            ctypes.c_float(config['scorethresh']),
                            ctypes.c_float(config['iouthresh']),
                            ctypes.byref(bboxes))

        boxes_per_image = []
        for i in range(0, nboxes*6, 6):
            boxes_per_image.append({'classid' : int(bboxes[i+4]),
                      'prob' : bboxes[i+5],
                        'll' : {'x' : bboxes[i+0],
                                'y' : bboxes[i+1]},
                         'ur' : {'x' : bboxes[i+2],
                                 'y' : bboxes[i+3]}})
        bbox_list_for_images.append(boxes_per_image) 
        _lib.clearBuffer(bboxes)
    return bbox_list_for_images

# fpga_Output : List of output arrays from FPGA, each item has batchSz dim
# config : dict of config params
# img_shape : list of Input image shapes, size = number of images
# biases : np.float32 vector, default value : bias_coco
def yolov2_postproc(fpga_output, config, img_shape, biases = yolov2_bias_coco):
    bbox_list_for_images = []
    nArrays = len(fpga_output)
    batchSz = len(img_shape)
    c_outputs = (PyArray * nArrays)()
    
    for bsz in range(batchSz):
        for i, arr in enumerate(fpga_output):
            batch = arr[bsz]
            x = PyArray(arr.shape[2], arr.shape[3], arr.shape[1], batch.ctypes.data_as(c_float_p))
            c_outputs[i] = x
        bboxes = c_float_p()
        nboxes = _lib.yolov2_postproc(ctypes.byref(c_outputs),
                            nArrays,
                            biases.ctypes.data_as(c_float_p),
                            config['net_h'],
                            config['net_w'],
                            config['classes'],
                            config['anchorCnt'],
                            int(img_shape[bsz][0]),
                            int(img_shape[bsz][1]),
                            ctypes.c_float(config['scorethresh']),
                            ctypes.c_float(config['iouthresh']),
                            ctypes.byref(bboxes))
        boxes_per_image = []
        for i in range(0, nboxes*6, 6):
            boxes_per_image.append({
                        'll' : {'x' : bboxes[i+0],
                                'y' : bboxes[i+1]},
                        'ur' : {'x' : bboxes[i+2],
                                 'y' : bboxes[i+3]},
                        'classid' : int(bboxes[i+4]),
                        'prob' : bboxes[i+5]})
        bbox_list_for_images.append(boxes_per_image) 
        _lib.clearBuffer(bboxes)
    return bbox_list_for_images

