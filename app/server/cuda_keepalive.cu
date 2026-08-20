#include "cuda_keepalive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace minitts::server {
namespace {

constexpr int kKeepaliveThreads = 256;

__global__ void cuda_keepalive_kernel(
    unsigned long long duration_cycles,
    unsigned int * heartbeat) {
    const unsigned long long started = clock64();
    unsigned int value = *heartbeat + threadIdx.x + 1;
    while (clock64() - started < duration_cycles) {
        value = value * 1664525u + 1013904223u;
    }
    atomicAdd(heartbeat, value | 1u);
}

void require_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA keepalive ") + operation + " failed: " + cudaGetErrorString(status));
    }
}

class CudaKeepaliveImpl final : public CudaKeepalive {
public:
    CudaKeepaliveImpl(int device, int interval_ms, int work_ms)
        : device_(device), interval_(interval_ms) {
        require_cuda(cudaSetDevice(device_), "cudaSetDevice");
        cudaDeviceProp properties{};
        require_cuda(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        duration_cycles_ = static_cast<unsigned long long>(properties.clockRate) * work_ms;
        int least_priority = 0;
        int greatest_priority = 0;
        require_cuda(
            cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
            "cudaDeviceGetStreamPriorityRange");
        (void)greatest_priority;
        require_cuda(
            cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, least_priority),
            "cudaStreamCreateWithPriority");
        try {
            require_cuda(cudaMalloc(&heartbeat_, sizeof(unsigned int)), "cudaMalloc");
            require_cuda(cudaMemset(heartbeat_, 0, sizeof(unsigned int)), "cudaMemset");
            worker_ = std::thread([this] { run(); });
        } catch (...) {
            if (heartbeat_ != nullptr) {
                (void)cudaFree(heartbeat_);
                heartbeat_ = nullptr;
            }
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
            throw;
        }
    }

    ~CudaKeepaliveImpl() override {
        stop_.store(true, std::memory_order_release);
        wake_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        (void)cudaSetDevice(device_);
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
        }
        if (heartbeat_ != nullptr) {
            (void)cudaFree(heartbeat_);
        }
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }

private:
    void run() noexcept {
        if (cudaSetDevice(device_) != cudaSuccess) {
            std::cerr << "CUDA keepalive could not select device " << device_ << "\n";
            return;
        }
        while (!stop_.load(std::memory_order_acquire)) {
            cuda_keepalive_kernel<<<1, kKeepaliveThreads, 0, stream_>>>(
                duration_cycles_,
                heartbeat_);
            const auto submit = cudaGetLastError();
            const auto complete = submit == cudaSuccess ? cudaStreamSynchronize(stream_) : submit;
            if (complete != cudaSuccess) {
                std::cerr << "CUDA keepalive stopped after an asynchronous error: "
                          << cudaGetErrorString(complete) << "\n";
                return;
            }
            std::unique_lock<std::mutex> lock(wait_mutex_);
            wake_.wait_for(lock, interval_, [this] {
                return stop_.load(std::memory_order_acquire);
            });
        }
    }

    int device_ = 0;
    std::chrono::milliseconds interval_;
    unsigned long long duration_cycles_ = 0;
    cudaStream_t stream_ = nullptr;
    unsigned int * heartbeat_ = nullptr;
    std::atomic<bool> stop_{false};
    std::mutex wait_mutex_;
    std::condition_variable wake_;
    std::thread worker_;
};

}  // namespace

std::unique_ptr<CudaKeepalive> start_cuda_keepalive(
    int device,
    int interval_ms,
    int work_ms) {
    if (interval_ms <= 0) {
        return nullptr;
    }
    auto keepalive = std::make_unique<CudaKeepaliveImpl>(device, interval_ms, work_ms);
    std::cout << "CUDA low-latency keepalive enabled on device " << device
              << " (" << work_ms << " ms work, " << interval_ms << " ms rest)"
              << std::endl;
    return keepalive;
}

}  // namespace minitts::server
