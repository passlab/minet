// Copyright 2019 Xilinx Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "dpu_test/dpu_test_runner.hpp"

#include <chrono>
#include <thread>

namespace vitis {
namespace ai {

DpuTestRunner::DpuTestRunner(float bias, const std::vector<std::int32_t> &dims)
    : bias_{bias}, job_counter_{0}, in0{"in0", dims, Tensor::DataType::FLOAT},
      in1{"in1", dims, Tensor::DataType::FLOAT},
      out{"out", dims, Tensor::DataType::FLOAT} {}

int calc(const float *in0, const float *in1, float *out, float bias,
         int length) {
  using namespace std::chrono_literals;
  for (auto i = 0; i < length; i++) {
    out[i] = in0[i] * in1[i] + bias;
  }
  std::this_thread::sleep_for(5s);
  return 0;
}

std::pair<uint32_t, int>
DpuTestRunner::execute_async(const std::vector<TensorBuffer *> &input,
                             const std::vector<TensorBuffer *> &output) {
  auto jobid = ++job_counter_;
  auto length = output[0]->get_tensor()->get_element_num();
  auto future_calc =
      std::async(std::launch::async, calc, (float *)input[0]->data().first,
                 (float *)input[1]->data().first,
                 (float *)output[0]->data().first, bias_, length);
  future_list_[jobid] = future_calc.share();
  return std::make_pair(jobid, 0);
}

int DpuTestRunner::wait(int jobid, int) {
  future_list_[jobid].wait();
  auto ret = future_list_[jobid].get();
  future_list_.erase(jobid);
  return ret;
}

DpuRunner::TensorFormat DpuTestRunner::get_tensor_format() {
  return DpuRunner::TensorFormat::NHWC;
}

std::vector<Tensor *> DpuTestRunner::get_input_tensors() {
  return {&in0, &in1};
}

std::vector<Tensor *> DpuTestRunner::get_output_tensors() { return {&out}; }
}
}
