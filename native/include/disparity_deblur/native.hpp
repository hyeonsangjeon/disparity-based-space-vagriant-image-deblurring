#pragma once

#include <cstddef>
#include <vector>

namespace disparity_deblur {

// Restore a contiguous HxWxC image with the canonical half-quadratic TV/L1 solver.
std::vector<double> tv_l1_deconvolve(
    const std::vector<double>& image,
    std::size_t height,
    std::size_t width,
    std::size_t channels,
    const std::vector<double>& kernel,
    std::size_t kernel_height,
    std::size_t kernel_width,
    double data_weight,
    double beta_max);

}  // namespace disparity_deblur
