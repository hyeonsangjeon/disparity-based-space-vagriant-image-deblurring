#include "disparity_deblur/native.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

py::array_t<double> deconvolve(
    py::array_t<double, py::array::c_style | py::array::forcecast> image,
    py::array_t<double, py::array::c_style | py::array::forcecast> kernel,
    double data_weight,
    double beta_max) {
  const py::buffer_info image_info = image.request();
  const py::buffer_info kernel_info = kernel.request();
  if (image_info.ndim != 3) {
    throw std::invalid_argument("native image must have shape HxWxC");
  }
  if (kernel_info.ndim != 2) {
    throw std::invalid_argument("native kernel must have shape HxW");
  }

  const auto height = static_cast<std::size_t>(image_info.shape[0]);
  const auto width = static_cast<std::size_t>(image_info.shape[1]);
  const auto channels = static_cast<std::size_t>(image_info.shape[2]);
  const auto kernel_height = static_cast<std::size_t>(kernel_info.shape[0]);
  const auto kernel_width = static_cast<std::size_t>(kernel_info.shape[1]);
  const auto* image_data = static_cast<const double*>(image_info.ptr);
  const auto* kernel_data = static_cast<const double*>(kernel_info.ptr);

  std::vector<double> input(
      image_data, image_data + height * width * channels);
  std::vector<double> psf(
      kernel_data, kernel_data + kernel_height * kernel_width);
  std::vector<double> output;
  {
    py::gil_scoped_release release;
    output = disparity_deblur::tv_l1_deconvolve(
        input,
        height,
        width,
        channels,
        psf,
        kernel_height,
        kernel_width,
        data_weight,
        beta_max);
  }

  py::array_t<double> result(py::array::ShapeContainer{
      static_cast<py::ssize_t>(height),
      static_cast<py::ssize_t>(width),
      static_cast<py::ssize_t>(channels),
  });
  py::buffer_info result_info = result.request();
  std::copy(output.begin(), output.end(), static_cast<double*>(result_info.ptr));
  return result;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() =
      "Optional portable C++17 TV/L1 deconvolution backend.";
  module.def(
      "tv_l1_deconvolve",
      &deconvolve,
      py::arg("image"),
      py::arg("kernel"),
      py::arg("data_weight"),
      py::arg("beta_max"),
      "Restore a padded HxWxC array with the native FFT solver.");
  module.def("implementation_info", []() {
    py::dict info;
    info["name"] = "portable-cpp17";
    info["fft"] = "radix-2/Bluestein";
    info["supports_rectangular"] = true;
    info["precision"] = "float64";
    return info;
  });
}
