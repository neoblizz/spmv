#pragma once

namespace util {

namespace device {

struct clock
{
  clock() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~clock() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() {
    cudaEventRecord(start_);
  }

  float milliseconds() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time;
  }

 private:
  cudaEvent_t start_, stop_;
};

} // namespace device
} // namespace util