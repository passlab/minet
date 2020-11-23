#pragma once

#include "dpu/dpu_runner.hpp"

#include <future>
#include <map>

namespace vitis {
namespace ai {

class DpuTestRunner : public DpuRunner {
public:
  explicit DpuTestRunner(float bias, const std::vector<std::int32_t> &dims);

  ~DpuTestRunner() = default;

  virtual std::pair<uint32_t, int>
  execute_async(const std::vector<TensorBuffer *> &input,
                const std::vector<TensorBuffer *> &output) override;

  virtual int wait(int jobid, int timeout) override;

  virtual TensorFormat get_tensor_format() override;

  virtual std::vector<Tensor *> get_input_tensors() override;

  virtual std::vector<Tensor *> get_output_tensors() override;

private:
  float bias_;
  uint32_t job_counter_;
  Tensor in0;
  Tensor in1;
  Tensor out;
  std::map<uint32_t, std::shared_future<int>> future_list_;
};
}
}
