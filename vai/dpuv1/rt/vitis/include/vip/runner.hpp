#pragma once

#include <cstdint>
#include <utility>
namespace vitis {

template <typename InputType, typename OutputType = InputType> class Runner {
public:
  virtual ~Runner() = default;

  /**
   * @brief execute_async
   *
   * @param in inputs with a customized type
   *
   * @param out outputs with a customized type
   *
   * @return pair<jodid, status> status 0 for exit successfully, others for
   * customized warnings or errors
   *
   */
  virtual std::pair<std::uint32_t, int> execute_async(InputType input,
                                                      OutputType output) = 0;

  /**
   * @brief wait
   *
   * @details modes: 1. Blocking wait for specific ID. 2. Non-blocking wait for
   * specific ID. 3. Blocking wait for any ID. 4. Non-blocking wait for any ID
   *
   * @param jobid job id, neg for any id, others for specific job id
   *
   * @param timeout timeout, neg for block for ever, 0 for non-block, pos for
   * block with a limitation(ms).
   *
   * @return status 0 for exit successfully, others for customized warnings or
   * errors
   *
   */
  virtual int wait(int jobid, int timeout = -1) = 0;
};
} // namespace vitis
