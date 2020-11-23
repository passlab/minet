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
ai_q_tensorflow quantize --input_frozen_graph ./rozen_graph.pb 
                                                --input_nodes "input_1" 
                                                --input_shapes ?,224,224,3 
                                                --output_nodes "conv2d_19/Relu" 
                                                --output_dir ../workspace/tf_chkpts/unet1/ 
                                                --input_fn graph_input_fn.calib_input 
                                                --calib_iter 10
                                                --gpu 0
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

