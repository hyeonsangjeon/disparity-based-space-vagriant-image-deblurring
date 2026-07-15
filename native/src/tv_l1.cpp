#include "disparity_deblur/native.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <future>
#include <stdexcept>
#include <vector>

namespace disparity_deblur {
namespace {

using Complex = std::complex<double>;

constexpr double kPi = 3.141592653589793238462643383279502884;

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void radix2_fft(std::vector<Complex>& values, bool inverse) {
  const std::size_t size = values.size();
  for (std::size_t index = 1, reversed = 0; index < size; ++index) {
    std::size_t bit = size >> 1;
    while ((reversed & bit) != 0) {
      reversed ^= bit;
      bit >>= 1;
    }
    reversed ^= bit;
    if (index < reversed) {
      std::swap(values[index], values[reversed]);
    }
  }

  for (std::size_t length = 2; length <= size; length <<= 1) {
    const double angle =
        (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(length);
    const Complex step(std::cos(angle), std::sin(angle));
    for (std::size_t start = 0; start < size; start += length) {
      Complex weight(1.0, 0.0);
      const std::size_t half = length >> 1;
      for (std::size_t offset = 0; offset < half; ++offset) {
        const Complex even = values[start + offset];
        const Complex odd = values[start + offset + half] * weight;
        values[start + offset] = even + odd;
        values[start + offset + half] = even - odd;
        weight *= step;
      }
    }
  }
  if (inverse) {
    const double scale = 1.0 / static_cast<double>(size);
    for (Complex& value : values) {
      value *= scale;
    }
  }
}

// Cache one axis's chirp and convolution spectrum instead of rebuilding
// Bluestein state for every row, column, channel, and solver iteration.
class FftPlan {
 public:
  explicit FftPlan(std::size_t size)
      : size_(size), power_of_two_(is_power_of_two(size)) {
    if (size_ < 1) {
      throw std::invalid_argument("FFT size must be positive");
    }
    if (power_of_two_ || size_ < 2) {
      return;
    }
    convolution_size_ = 1;
    while (convolution_size_ < size_ * 2 - 1) {
      convolution_size_ <<= 1;
    }
    chirp_.resize(size_);
    convolution_fft_.assign(convolution_size_, Complex(0.0, 0.0));
    for (std::size_t index = 0; index < size_; ++index) {
      const double index_value = static_cast<double>(index);
      const double angle =
          kPi * index_value * index_value / static_cast<double>(size_);
      chirp_[index] = Complex(std::cos(angle), -std::sin(angle));
      const Complex reverse_chirp = std::conj(chirp_[index]);
      convolution_fft_[index] = reverse_chirp;
      if (index != 0) {
        convolution_fft_[convolution_size_ - index] = reverse_chirp;
      }
    }
    radix2_fft(convolution_fft_, false);
  }

  std::size_t workspace_size() const {
    return convolution_size_;
  }

  void transform(
      std::vector<Complex>& values,
      bool inverse,
      std::vector<Complex>& workspace) const {
    if (values.size() != size_) {
      throw std::invalid_argument("FFT input does not match its plan");
    }
    if (!inverse) {
      forward(values, workspace);
      return;
    }
    for (Complex& value : values) {
      value = std::conj(value);
    }
    forward(values, workspace);
    const double scale = 1.0 / static_cast<double>(size_);
    for (Complex& value : values) {
      value = std::conj(value) * scale;
    }
  }

 private:
  void forward(
      std::vector<Complex>& values,
      std::vector<Complex>& workspace) const {
    if (size_ < 2) {
      return;
    }
    if (power_of_two_) {
      radix2_fft(values, false);
      return;
    }
    if (workspace.size() != convolution_size_) {
      workspace.resize(convolution_size_);
    }
    std::fill(
        workspace.begin(), workspace.end(), Complex(0.0, 0.0));
    for (std::size_t index = 0; index < size_; ++index) {
      workspace[index] = values[index] * chirp_[index];
    }
    radix2_fft(workspace, false);
    for (std::size_t index = 0; index < convolution_size_; ++index) {
      workspace[index] *= convolution_fft_[index];
    }
    radix2_fft(workspace, true);
    for (std::size_t index = 0; index < size_; ++index) {
      values[index] = workspace[index] * chirp_[index];
    }
  }

  std::size_t size_;
  bool power_of_two_;
  std::size_t convolution_size_ = 0;
  std::vector<Complex> chirp_;
  std::vector<Complex> convolution_fft_;
};

void fft2(
    std::vector<Complex>& values,
    std::size_t height,
    std::size_t width,
    bool inverse,
    const FftPlan& height_plan,
    const FftPlan& width_plan) {
  // Separable row/column transforms preserve arbitrary rectangular dimensions.
  std::vector<Complex> line(std::max(height, width));
  std::vector<Complex> width_workspace(width_plan.workspace_size());
  std::vector<Complex> height_workspace(height_plan.workspace_size());
  for (std::size_t y = 0; y < height; ++y) {
    line.resize(width);
    for (std::size_t x = 0; x < width; ++x) {
      line[x] = values[y * width + x];
    }
    width_plan.transform(line, inverse, width_workspace);
    for (std::size_t x = 0; x < width; ++x) {
      values[y * width + x] = line[x];
    }
  }
  for (std::size_t x = 0; x < width; ++x) {
    line.resize(height);
    for (std::size_t y = 0; y < height; ++y) {
      line[y] = values[y * width + x];
    }
    height_plan.transform(line, inverse, height_workspace);
    for (std::size_t y = 0; y < height; ++y) {
      values[y * width + x] = line[y];
    }
  }
}

std::vector<Complex> psf_to_otf(
    const std::vector<double>& kernel,
    std::size_t kernel_height,
    std::size_t kernel_width,
    std::size_t height,
    std::size_t width,
    const FftPlan& height_plan,
    const FftPlan& width_plan) {
  // Match NumPy's top-left embedding followed by a negative center roll.
  std::vector<Complex> padded(height * width);
  const std::size_t center_y = kernel_height / 2;
  const std::size_t center_x = kernel_width / 2;
  for (std::size_t y = 0; y < kernel_height; ++y) {
    for (std::size_t x = 0; x < kernel_width; ++x) {
      const std::size_t target_y = (y + height - center_y) % height;
      const std::size_t target_x = (x + width - center_x) % width;
      padded[target_y * width + target_x] =
          kernel[y * kernel_width + x];
    }
  }
  fft2(
      padded, height, width, false, height_plan, width_plan);
  return padded;
}

double soft_threshold(double value, double threshold) {
  if (value > threshold) {
    return value - threshold;
  }
  if (value < -threshold) {
    return value + threshold;
  }
  return 0.0;
}

}  // namespace

std::vector<double> tv_l1_deconvolve(
    const std::vector<double>& image,
    std::size_t height,
    std::size_t width,
    std::size_t channels,
    const std::vector<double>& kernel,
    std::size_t kernel_height,
    std::size_t kernel_width,
    double data_weight,
    double beta_max) {
  // FFT plans and derivative spectra are shared by every image channel.
  if (height < 2 || width < 2 || channels < 1) {
    throw std::invalid_argument(
        "native image dimensions must be H>=2, W>=2, C>=1");
  }
  if (kernel_height < 1 || kernel_width < 1 ||
      kernel_height > height || kernel_width > width) {
    throw std::invalid_argument("native kernel dimensions are invalid");
  }
  if (image.size() != height * width * channels ||
      kernel.size() != kernel_height * kernel_width) {
    throw std::invalid_argument("native input buffer size mismatch");
  }
  if (data_weight <= 0.0 || beta_max < 1.0) {
    throw std::invalid_argument("native solver parameters are invalid");
  }

  const std::size_t pixel_count = height * width;
  const FftPlan height_plan(height);
  const FftPlan width_plan(width);
  const std::vector<Complex> kernel_otf = psf_to_otf(
      kernel,
      kernel_height,
      kernel_width,
      height,
      width,
      height_plan,
      width_plan);
  std::vector<Complex> dx_otf(pixel_count);
  std::vector<Complex> dy_otf(pixel_count);
  dx_otf[0] = -1.0;
  dx_otf[1] = 1.0;
  dy_otf[0] = -1.0;
  dy_otf[width] = 1.0;
  fft2(
      dx_otf, height, width, false, height_plan, width_plan);
  fft2(
      dy_otf, height, width, false, height_plan, width_plan);

  std::vector<double> kernel_power(pixel_count);
  std::vector<double> derivative_power(pixel_count);
  for (std::size_t index = 0; index < pixel_count; ++index) {
    kernel_power[index] = std::norm(kernel_otf[index]);
    derivative_power[index] =
        std::norm(dx_otf[index]) + std::norm(dy_otf[index]);
  }

  std::vector<double> output(image.size());
  std::vector<std::vector<double>> channel_outputs(
      channels, std::vector<double>(pixel_count));
  const auto process_channel = [&](std::size_t channel) {
    std::vector<double> current(pixel_count);
    std::vector<Complex> observation_fft(pixel_count);
    std::vector<Complex> auxiliary_x(pixel_count);
    std::vector<Complex> auxiliary_y(pixel_count);
    std::vector<Complex> solution_fft(pixel_count);
    for (std::size_t index = 0; index < pixel_count; ++index) {
      current[index] = image[index * channels + channel];
      observation_fft[index] = current[index];
    }
    fft2(
        observation_fft, height, width, false, height_plan, width_plan);

    double beta = 1.0;
    while (beta <= beta_max + 1e-12) {
      const double threshold = 1.0 / beta;
      for (std::size_t y = 0; y < height; ++y) {
        const std::size_t next_y = (y + 1) % height;
        for (std::size_t x = 0; x < width; ++x) {
          const std::size_t next_x = (x + 1) % width;
          const std::size_t index = y * width + x;
          auxiliary_x[index] = soft_threshold(
              current[y * width + next_x] - current[index], threshold);
          auxiliary_y[index] = soft_threshold(
              current[next_y * width + x] - current[index], threshold);
        }
      }
      fft2(
          auxiliary_x, height, width, false, height_plan, width_plan);
      fft2(
          auxiliary_y, height, width, false, height_plan, width_plan);
      for (std::size_t index = 0; index < pixel_count; ++index) {
        const Complex numerator =
            data_weight * std::conj(kernel_otf[index]) *
                observation_fft[index] +
            beta *
                (std::conj(dx_otf[index]) * auxiliary_x[index] +
                 std::conj(dy_otf[index]) * auxiliary_y[index]);
        const double denominator =
            data_weight * kernel_power[index] +
            beta * derivative_power[index] + 1e-8;
        solution_fft[index] = numerator / denominator;
      }
      fft2(
          solution_fft, height, width, true, height_plan, width_plan);
      for (std::size_t index = 0; index < pixel_count; ++index) {
        current[index] = solution_fft[index].real();
      }
      beta *= 2.0;
    }
    for (std::size_t index = 0; index < pixel_count; ++index) {
      channel_outputs[channel][index] =
          std::clamp(current[index], 0.0, 1.0);
    }
  };

  if (channels > 1 && pixel_count >= 16384) {
    std::vector<std::future<void>> workers;
    workers.reserve(channels);
    for (std::size_t channel = 0; channel < channels; ++channel) {
      workers.push_back(std::async(
          std::launch::async, process_channel, channel));
    }
    for (auto& worker : workers) {
      worker.get();
    }
  } else {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      process_channel(channel);
    }
  }
  for (std::size_t index = 0; index < pixel_count; ++index) {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      output[index * channels + channel] =
          channel_outputs[channel][index];
    }
  }
  return output;
}

}  // namespace disparity_deblur
