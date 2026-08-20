#pragma once

#include <memory>

namespace minitts::server {

class CudaKeepalive {
public:
    virtual ~CudaKeepalive() = default;
};

// Submits a tiny low-priority CUDA operation at the requested interval. This
// is intentionally opt-in: it prevents aggressive WDDM idle downclocking for
// latency-critical resident servers, at the cost of higher idle GPU power.
std::unique_ptr<CudaKeepalive> start_cuda_keepalive(
    int device,
    int interval_ms,
    int work_ms);

}  // namespace minitts::server
