#!/usr/bin/env python
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
from __future__ import print_function
import argparse
import json # parse compiler
import google.protobuf.text_format as pbtf
import caffe.proto.caffe_pb2 as caffe_pb2
from vai.dpuv1.rt import xdnn_io

def build_tensor_list(netpb):
  tensorlist = {}
  for l in netpb.layer:
    for t in l.top:
      print("Adding output %s"%t)
      tensorlist[t] = [l.name]
    for b in l.bottom:
      if b in tensorlist:
        tensorlist[b].append(l.name)
      else:
        tensorlist[b] = [l.name]
  print("Done")
  for k in tensorlist.keys():
    print("Key:",k,"Source: %s"%tensorlist[k][0]," Sinks:",tensorlist[k][1:])
  return tensorlist

def cut_subgraph(netpb, in_blob, graphname, args):
  outpb = caffe_pb2.NetParameter()
  visited = set()
  tensorlist = build_tensor_list(netpb)

  with open(args["netcfg"]) as f:
    compilerData = json.load(f)

  FPGA_input_layer = []
  # get input list to FPGA layer
  inputs=compilerData['inputs']
  if (inputs == None):
    print("ERROR: Can't find %s in compiler JSON file"%'inputs')
    return None
  for inputItem in inputs:
    input_name=inputItem['input_name']
    FPGA_input_layer.append(input_name)
    print("Found input to FPGA layer %s"%input_name)


  # add merged list to FPGA layer
  network=compilerData['network']
  networkMergeList=[]
  if (network == None):
    print("ERROR: Can't find %s in compiler JSON file"%'network')
    return None
  for networkItem in network:
    if 'merged' in networkItem:
        merged=networkItem['merged']
        if (merged == None or len(merged) == 0) and (networkItem['layer'] != "data movement"):
          print("ERROR: Empty merge list found in non data movement layer")
          return None
        networkMergeList.append(merged)
        for mergeItem in merged:
          visited.add(mergeItem)

  compilerOutputs = compilerData['outputs']
  compilerOutputsDict = {}
  outTensorNames = set()
  for o in compilerOutputs:
    outTensorNames.add(o["previous_tensors"][0])
    compilerOutputsDict[o["previous_layers"][0]] = (o["previous_tensors"][0]);

  # create new prototxt
  newpb_layer = []

  #Add input
  input_layer = []
  input_start_flag = True
  for l in netpb.layer:
    # do not remove input layers if in visited
    #if (l.type == 'Input' and l.name in visited):
    if (input_start_flag):
      input_layer.append(l)
      for in_layer in FPGA_input_layer:
          if in_layer == l.name:
              input_start_flag = False

  newpb_layer.extend(input_layer)


  # add Custom Layer
  pylayer = caffe_pb2.LayerParameter()
  pylayer.name = "vai/dpuv1/%s"%(graphname)
  pylayer.type = "Python"
  pylayer.bottom.extend(FPGA_input_layer)
  for o in outTensorNames:
    pylayer.top.append ( o )
  #if args["profile"]:
  #  pylayer.top.append("vai/dpuv1/"+graphname+"/latency")

  pylayer.python_param.module = "vai.dpuv1.rt.scripts.framework.caffe.CaffeXFDNN"
  pylayer.python_param.layer = "CaffeXFDNN"

  input_names = []
  input_names.extend(FPGA_input_layer)
  args["input_names"] = input_names
  args["output_names"] = list(outTensorNames)
  pylayer.python_param.param_str = str(args)
  newpb_layer.append(pylayer)

  #Add non custom layer (Shell layer)
  layer_exist = False
  newpb_shell_layer = []
  for l in netpb.layer:
    # do remove layers if in visited
    if l.name not in visited :
        for i in newpb_layer:
            if l.name == i.name:
                layer_exist = True
                break
        if not layer_exist :
            newpb_shell_layer.append(l)

        layer_exist = False

  newpb_layer.extend(newpb_shell_layer)


  #ONEHACK TO FIX MERGE LAYER TENSOR NAMES
  newpb_shell_layer.extend(input_layer)
  # ensure connection within shell layers are present
  for l in newpb_shell_layer:
    for b in l.bottom:
      for layerName in tensorlist[b]:
        if layerName in visited:
          # We found a connection from the custom layer to the shell
          print("INFO: Layer %s is not in shell"%layerName)
          if b not in args["output_names"]:
            # The tensor name does not exist as output of custom layer
            # Do the ONEHACK
            # add correct blob name
            # get intersection of layerName and output previous_layers
            # add output previous_tensors
            # Note may add more than one tensor, if it does, then there is an issue with the compiler
            for key, value in compilerOutputsDict.items():
              for merged in networkMergeList:
                if (layerName in merged) and (key in merged):
                  # remove blob name
                  try:
                    l.bottom.remove(b)
                    l.bottom.append(value)
                    print("INFO: Blob %s does not exist as output of custom layer"%b)
                  except:
                    pass

  outpb.layer.extend(newpb_layer)
  return outpb

# If the user provides a train_val prototxt for running accuracy, we can hijack the data layers, and the accuracy layers
# They will be prepended, and postpended respectively
def add_accuracy(netpb, trainpb):
  outpb = caffe_pb2.NetParameter()
  outpb.layer.extend([l for l in trainpb.layer if l.type in ["Data","ImageData"]])
  outpb.layer.extend([l for l in netpb.layer if l.type not in ["Input"]])
  outpb.layer.extend([l for l in trainpb.layer if l.type in ["Accuracy"]])
  return outpb

def default_parser():
  parser = argparse.ArgumentParser(description='pyXDNN')
  parser.add_argument('--xclbin', help='.xclbin file', required=True, type=xdnn_io.extant_file, metavar="FILE")
  parser.add_argument('--netcfg', help='FPGA instructions generated by compiler for the network',
                      required=True, type=xdnn_io.extant_file, metavar="FILE")
  parser.add_argument('--quantizecfg', help="Network's quantization parameters file",
                      required=True, type=xdnn_io.extant_file, metavar="FILE")
  parser.add_argument('--weights', help="Folder path to network parameters/weights",
      required=True, type=xdnn_io.extant_file, metavar="FILE")
  parser.add_argument('--inproto', default="", help='Original graph')
  parser.add_argument("--outproto", default = "", required = True )
  parser.add_argument('--trainproto', default=None, help='Original training graph')
  parser.add_argument("--outtrainproto", default = None )
  parser.add_argument('--batch_sz', type=xdnn_io.max_batch_size, default=-1, help='batch size')
  #parser.add_argument('--profile', type=bool, default=False, help='batch size')
  parser.add_argument('--cutAfter', default="", help='Node in graph to start cutting after')
  return parser

if __name__ == "__main__":
  # Run

  parser = default_parser()

  args = parser.parse_args()

  args = xdnn_io.make_dict_args(args)

  netpb = caffe_pb2.NetParameter()

  with open(args["inproto"],"r") as f:
    pbtf.Parse(f.read(), netpb)

  srctensor1 = args["cutAfter"]
  outpb = cut_subgraph(netpb, srctensor1, "subgraph0", args)

  with open(args["outproto"],"w") as f:
    f.write(str(outpb))

  # if user passes the train_val prototxt, we will steal the data and accuracy layers
  if(args["trainproto"]):
    trainpb = caffe_pb2.NetParameter()
    with open(args["trainproto"],"r") as f:
      pbtf.Parse(f.read(), trainpb)
    outpb = add_accuracy(outpb,trainpb)
  if(args["outtrainproto"]):
    with open(args["outtrainproto"],"w") as f:
      f.write(str(outpb))

class CaffeCutter():

  def __init__(self,**kwargs):
    arglist = []
    for k,v in kwargs.items():
      arglist.append("--"+str(k))
      arglist.append(str(v))
      print (arglist)
    parser = default_parser()
    args = parser.parse_args(arglist)
    self.args = xdnn_io.make_dict_args(args)

  def cut(self):
    print (self.args)
    args = self.args
    netpb = caffe_pb2.NetParameter()

    with open(args["inproto"],"r") as f:
      pbtf.Parse(f.read(), netpb)

    srctensor1 = args["cutAfter"]
    outpb = cut_subgraph(netpb, srctensor1, "subgraph0", args)

    with open(args["outproto"],"w") as f:
      f.write(str(outpb))

    # if user passes the train_val prototxt, we will steal the data and accuracy layers
    if(args["trainproto"]):
      trainpb = caffe_pb2.NetParameter()
      with open(args["trainproto"],"r") as f:
        pbtf.Parse(f.read(), trainpb)
      outpb = add_accuracy(outpb,trainpb)
    if(args["outtrainproto"]):
      with open(args["outtrainproto"],"w") as f:
        f.write(str(outpb))
