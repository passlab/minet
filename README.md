# The step to use Vitis-AI

## Prepare Vitis-AI Docker and virtual enviorment

You can download the lastest docker from Xilinx Vitis-AI docker hub or build from their repo.

If you use the pre-build docker, it can be download from https://hub.docker.com/r/xilinx/vitis-ai/tags.

If you want to build you own docker, you can use following command:

```
git clone https://github.com/Xilinx/Vitis-AI.git

cd Vitis-AI

cd docker

./docker_build_gpu.sh

```
Those command lines will build a docker for you. Then you can start the docker with following command:

```
./docker_run.sh xilinx/vitis-ai-gpu:latest
```

After you enter Vitis-AI docker container, you need to activate the virtual enviorment for Machine learning framework such as Tensorflow, Caffe, pytorch and so on. An example shows as follow :

```
conda activate vitis-ai-tensorflow
```

When you enter the Virtual enviorment, you may need source the seting for several packages such as: xilinx xbuter, python, tensorflow and so on.
```
source /workspace/alveo/overlaybins/setup.sh
```

## Use xilinx tools

1. Prepare your network. You can train your own network by any machine learning framework or download vitis-ai model from their model zoo from https://github.com/Xilinx/AI-Model-Zoo 

2. If you use keras to train you model, you need transfer your model to tensorflow checkpoint files with command: 
  ```
  python keras2TF.py 
  or
  python3 keras2TF.py
  ```
You can find this keras2TF code in our docker images on carina. The input of this python program is keras model, and output should be four files: infer_graph.pb, model.ckpt.medta, checkpoint and model.ckpt.000001.

3. After you have your tensorflow check point files, you need freeze the the inference graph for Vitis-AI quantized.  An example shows as follow :
  ```
freeze_graph --input_graph infer_graph.pb 
                         --input_checkpoint float_model.ckpt 
                         --input_binary      true 
                         --output_graph frozen_graph.pb 
                         --output_node_names conv2d_24/Sigmoid

  ```
  This freeze_graph tool is come with the docker continer. You can also manually freeze your ckpt files with our code which locat at tf-ckpt/unet1/test.py. This step will generate freeze graph for TensorFlow with the infer_graph we generate in last step. The generated files is a pb format file named frozen_graph.pb which is used for quantize.
  
4. Evaluate the original graph(optional) with command:
  ```
python eval_graph.py --graph= ./graph.pb  
                     --input_node="input_1" 
                     --output_node="conv2d_19/Relu" 
                     --gpu=0

  ```
  This is optional step for evaluating the graph we generated in last step and make sure the network structure is correct.
  
 5. Quantized the model with Vits-AI tools, an example shows as follow:
 
  ```
ai_q_tensorflow quantize --input_frozen_graph ./rozen_graph.pb 
                                                --input_nodes "input_1" 
                                                --input_shapes ?,224,224,3 
                                                --output_nodes "conv2d_19/Relu" 
                                                --output_dir ../workspace/tf_chkpts/unet1/ 
                                                --input_fn graph_input_fn.calib_input 
                                                --calib_iter 10
                                                --gpu 0
  ```

  This step is used for quantized the model, reduce the weight and change the unimportant weight precision with Xilinx tools vai_q_tensorflow. It will generate two files: quantize_eval_model.pb and deploy_model.pb. If you use different machine learning framework such as caffe, you can use vai_q_caffe.
    
 6. Evaluate the quantized graph(optional) with command:
 
```
python eval_quantized_graph.py --graph=./ uantize_eval_model.pb 
                                                            --input_node="input_1" 
                                                            --output_node="conv2d_19/Relu" 
                                                            --gpu=0

 ```
 This is optional step for evaluate the quantized graph. Make sure they precision and the accuracy of your network will not significant reduced after quantized.
    
7. Compile the model with command:
  
```
vai_c_tensorflow --frozen_pb deploy_model.pb 
                               --arch /opt/vitis_ai/compiler/arch/DPUCADX8G/ALVEO/arch.json  
                               --output_dir  ./unet1 
                               --options    "{'mode':'normal'}" 
                               --net_name unet1
```

This step will generate different files according to the arch you use. For DPUCADX8G (DPU v1), it should generate weight.h5, quantizer.json, compiler.json and meta.json. However, after investage into their source code which is loacted in "/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/vai_c_tensorflow", this code can only generate a quantized info file.
    
 8. Generate the weight.h5, quantizer.json, compiler.json and meta.json. (With error, working on it.)
  
```
python vaicompiler.py

```
After investage, we finally find the code which used for gerenate the model file. However, this code has some issues with transfer the model file into Hareware code.
In this step, we also modify the code in "/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py" and code in "/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/utils/xdnn_rt_tf.py(xdnn_rt_base.py)". For now, it still does not works.

The laste file that I modified is vaic/dpuv1/bin/xfdnn_compiler_tensorflow.py. From line 131 to 194, the try generated code does not works.
    
The error messages:

```
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
Please specify a quantization file
Please specify the input shapes
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py:36: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

GenerateCode: /workspace/tutorial/Vitis-AI-Tutorials/files/workspace/tf_chkpts/unet1/unet1
Weights: None
PngFile: None
ConcatStrategy: None
Strategy: all
ScheduleFile: None
DDR: 256
DSP: 96
Verbose: False
FromTF: True
Memory: 9
Byte per Pixel: 1
Start compiling

**************************************************
* BUILDING DATA FLOW GRAPH
**************************************************
Reading pre-build graph

######### load_graph arguments #############
networkfile               deploy_model.pb
loadmode                  binary
startnode                 None
finalnode                 None
inclusive                 False
batch_sz                  None
fixinputnames             None
placeholdershape          None
remove_training_nodes     None
remove_redundant_nodes    None
freeze_blacklist          None
freeze_whitelist          None
graph_savepath            None
#############################################

WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py:345: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.gfile.GFile.
freeze model
.... node count 51
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py:247: remove_training_nodes (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.remove_training_nodes`
.... node count after removing training nodes 51
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py:251: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
.... node count after removing redundant nodes 51
.... node count after removing blacklisted nodes 51
NodeDef mentions attr 'opos' not in Op<name=Placeholder; signature= -> output:dtype; attr=dtype:type; attr=shape:shape,default=<unknown>>; NodeDef: {{node input_1}}. (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
<class 'tensorflow.core.framework.graph_pb2.GraphDef'>
input_1
input_1 [] list {
  i: 8
  i: 6
}
 [] []
1 input_1 8 6

conv2d_1/convolution
conv2d_1/convolution list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}

2 conv2d_1/convolution 8 6 8 4 8 5 8 8

activation_1/Relu
activation_1/Relu list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
3 activation_1/Relu 8 4 8 4

conv2d_2/convolution
conv2d_2/convolution list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 6
}

4 conv2d_2/convolution 8 4 8 4 8 9 8 6

activation_2/Relu
activation_2/Relu list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
5 activation_2/Relu 8 4 8 4

max_pooling2d_1/MaxPool
max_pooling2d_1/MaxPool list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
6 max_pooling2d_1/MaxPool 8 4 8 4

conv2d_3/convolution
conv2d_3/convolution list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

7 conv2d_3/convolution 8 4 8 5 8 9 8 5

activation_3/Relu
activation_3/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
8 activation_3/Relu 8 5 8 5

conv2d_4/convolution
conv2d_4/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

9 conv2d_4/convolution 8 5 8 5 8 9 8 5

activation_4/Relu
activation_4/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
10 activation_4/Relu 8 5 8 5

max_pooling2d_2/MaxPool
max_pooling2d_2/MaxPool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
11 max_pooling2d_2/MaxPool 8 5 8 5

conv2d_5/convolution
conv2d_5/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 5
}

12 conv2d_5/convolution 8 5 8 5 8 10 8 5

activation_5/Relu
activation_5/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
13 activation_5/Relu 8 5 8 5

conv2d_6/convolution
conv2d_6/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 5
}

14 conv2d_6/convolution 8 5 8 5 8 10 8 5

activation_6/Relu
activation_6/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
15 activation_6/Relu 8 5 8 5

max_pooling2d_3/MaxPool
max_pooling2d_3/MaxPool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
16 max_pooling2d_3/MaxPool 8 5 8 5

conv2d_7/convolution
conv2d_7/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 5
}

17 conv2d_7/convolution 8 5 8 5 8 10 8 5

activation_7/Relu
activation_7/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
18 activation_7/Relu 8 5 8 5

conv2d_8/convolution
conv2d_8/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 11
}
 list {
  i: 8
  i: 5
}

19 conv2d_8/convolution 8 5 8 5 8 11 8 5

activation_8/Relu
activation_8/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
20 activation_8/Relu 8 5 8 5

max_pooling2d_4/MaxPool
max_pooling2d_4/MaxPool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
21 max_pooling2d_4/MaxPool 8 5 8 5

conv2d_9/convolution
conv2d_9/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 11
}
 list {
  i: 8
  i: 5
}

22 conv2d_9/convolution 8 5 8 5 8 11 8 5

activation_9/Relu
activation_9/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
23 activation_9/Relu 8 5 8 5

conv2d_10/convolution
conv2d_10/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 11
}
 list {
  i: 8
  i: 5
}

24 conv2d_10/convolution 8 5 8 5 8 11 8 5

activation_10/Relu
activation_10/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
25 activation_10/Relu 8 5 8 5

up_sampling2d_1/ResizeBilinear
up_sampling2d_1/ResizeBilinear list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
26 up_sampling2d_1/ResizeBilinear 8 5 8 5

concatenate_1/concat
concatenate_1/concat list {
  i: 8
  i: 5
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
27 concatenate_1/concat 8 5 8 5 8 5

conv2d_11/convolution
conv2d_11/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 12
}
 list {
  i: 8
  i: 5
}

28 conv2d_11/convolution 8 5 8 5 8 12 8 5

activation_11/Relu
activation_11/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
29 activation_11/Relu 8 5 8 5

conv2d_12/convolution
conv2d_12/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 6
}

30 conv2d_12/convolution 8 5 8 5 8 10 8 6

activation_12/Relu
activation_12/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
31 activation_12/Relu 8 5 8 5

up_sampling2d_2/ResizeBilinear
up_sampling2d_2/ResizeBilinear list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
32 up_sampling2d_2/ResizeBilinear 8 5 8 5

concatenate_2/concat
concatenate_2/concat list {
  i: 8
  i: 5
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
33 concatenate_2/concat 8 5 8 5 8 5

conv2d_13/convolution
conv2d_13/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 11
}
 list {
  i: 8
  i: 5
}

34 conv2d_13/convolution 8 5 8 5 8 11 8 5

activation_13/Relu
activation_13/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
35 activation_13/Relu 8 5 8 5

conv2d_14/convolution
conv2d_14/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 6
}

36 conv2d_14/convolution 8 5 8 5 8 10 8 6

activation_14/Relu
activation_14/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
37 activation_14/Relu 8 5 8 5

up_sampling2d_3/ResizeBilinear
up_sampling2d_3/ResizeBilinear list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
38 up_sampling2d_3/ResizeBilinear 8 5 8 5

concatenate_3/concat
concatenate_3/concat list {
  i: 8
  i: 5
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
39 concatenate_3/concat 8 5 8 5 8 5

conv2d_15/convolution
conv2d_15/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 11
}
 list {
  i: 8
  i: 6
}

40 conv2d_15/convolution 8 5 8 5 8 11 8 6

activation_15/Relu
activation_15/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
41 activation_15/Relu 8 5 8 5

conv2d_16/convolution
conv2d_16/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 6
}

42 conv2d_16/convolution 8 5 8 4 8 10 8 6

activation_16/Relu
activation_16/Relu list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
43 activation_16/Relu 8 4 8 4

up_sampling2d_4/ResizeBilinear
up_sampling2d_4/ResizeBilinear list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
44 up_sampling2d_4/ResizeBilinear 8 4 8 4

concatenate_4/concat
concatenate_4/concat list {
  i: 8
  i: 4
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
45 concatenate_4/concat 8 4 8 4 8 4

conv2d_17/convolution
conv2d_17/convolution list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 11
}
 list {
  i: 8
  i: 6
}

46 conv2d_17/convolution 8 4 8 5 8 11 8 6

activation_17/Relu
activation_17/Relu list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
47 activation_17/Relu 8 5 8 5

conv2d_18/convolution
conv2d_18/convolution list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 6
}

48 conv2d_18/convolution 8 5 8 4 8 10 8 6

activation_18/Relu
activation_18/Relu list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
49 activation_18/Relu 8 4 8 4

conv2d_19/convolution
conv2d_19/convolution list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 1
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 9
}

50 conv2d_19/convolution 8 4 8 1 8 8 8 9

conv2d_19/Relu
conv2d_19/Relu list {
  i: 8
  i: 1
}
 list {
  i: 8
  i: 1
}
 [] []
51 conv2d_19/Relu 8 1 8 1

Thank you, we wrote the deephi quantization file /workspace/tutorial/Vitis-AI-Tutorials/files/workspace/tf_chkpts/unet1/unet1_fix.txt
Traceback (most recent call last):
  File "/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/vai_c_tensorflow", line 202, in <module>
    compiler.compile()
  File "/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/vai_c_tensorflow", line 176, in compile
    os.rename(DIR+weight,    DIR+'weights.h5'    )
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/tutorial/Vitis-AI-Tutorials/files/workspace/tf_chkpts/unet1/deploy_model.pb_data.h5' -> '/workspace/tutorial/Vitis-AI-Tutorials/files/workspace/tf_chkpts/unet1/weights.h5'
(vitis-ai-tensorflow) yshi10@cci-carina:/workspace/tutorial/Vitis-AI-Tutorials/files/workspace/tf_chkpts/unet1$ 

  9. Build the code with command:

```
cd workspace
make
```
In this step, we used the model which generated by last steps(model with errors) and used unified vitis-ai APIs to run on alveo platform to do the inference. It will generate the binary files in build folder.
    
  10. Run the infer
  
```
source run.sh
```
In this step, it should talk with Xilinx card and do the inference. But it encounters the error with preivous model.

