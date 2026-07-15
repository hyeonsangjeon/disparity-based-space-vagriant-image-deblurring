# Optional C++ and CUDA acceleration

The default `numpy` backend remains the canonical, portable reference implementation.
Acceleration is explicit and optional: selecting an unavailable backend raises an error
instead of silently changing algorithms.

| Backend | Installation | Intended benefit | Important limitation |
| --- | --- | --- | --- |
| `numpy` | Included by default | Reproducible baseline with optimized NumPy FFT | CPU execution |
| `native-cpp` | Build from this source checkout | C++17 embedding, GIL-free execution, rectangular radix-2/Bluestein FFT, and a native extension point | Not universally faster than NumPy, especially when reflection padding creates non-power-of-two dimensions |
| `cuda` | Install CuPy matching the local CUDA runtime | Keeps each TV/L1 iteration on the GPU and uses cuFFT for large images and repeated regional restoration | Requires an NVIDIA GPU, CUDA runtime, and compatible CuPy wheel |

## Build the optional C++ backend

The extension has no external FFT dependency. It uses a portable C++17 radix-2 FFT and
Bluestein's algorithm for arbitrary portrait, landscape, and odd-sized dimensions.

```sh
uv sync --group native
uv run --group native python tools/build_native.py
uv run python -c "from disparity_deblur import backend_status; print(backend_status())"
```

Run the pipeline explicitly:

```sh
uv run disparity-deblur \
  --blurred path/to/blurred.png \
  --noisy path/to/noisy.png \
  --output-dir output/native \
  --backend native-cpp
```

The extension uses float64 and is tested against the NumPy output on square, rectangular,
and odd-sized inputs. Small floating-point differences are expected. The portable backend
can outperform NumPy for favorable power-of-two FFT sizes, but reflection padding often
creates non-power-of-two dimensions where NumPy may remain faster. Measure the real target
images rather than assuming a speedup.

## Enable the CUDA backend

Install the CuPy package matching the machine's CUDA runtime. For CUDA 12, for example:

```sh
uv pip install cupy-cuda12x
uv run python -c "from disparity_deblur import backend_status; print(backend_status())"
```

Then select CUDA explicitly:

```sh
uv run disparity-deblur \
  --blurred path/to/blurred.png \
  --noisy path/to/noisy.png \
  --output-dir output/cuda \
  --backend cuda
```

The implementation transfers each unique regional restoration to the GPU, retains all
half-quadratic iterations and FFT operations there, and transfers the final result back
for normalized region merging. Benefits are most likely for large frames, multiple unique
regional PSFs, and repeated runs where CUDA initialization and FFT planning are amortized.
The first run should be treated as warm-up.

## Compare installed backends

Use the deterministic benchmark tool on the intended image dimensions and padding:

```sh
uv run --group native python tools/benchmark_backends.py \
  --height 512 \
  --width 768 \
  --kernel-size 17 \
  --beta-max 8 \
  --repeats 5
```

The JSON report includes median runtime and maximum absolute error relative to NumPy.
Do not publish a speedup claim without recording the CPU/GPU model, CUDA and CuPy versions,
image dimensions, PSF count, padding, warm-up policy, and repeat count.

## Reproducibility policy

- Public benchmark artifacts continue to use `backend: numpy` unless a manifest explicitly
  records another backend.
- Optional backends do not change registration, segmentation, PSF estimation, region
  masks, or normalized feather merging.
- CUDA availability is checked before processing; there is no automatic fallback.
- CI builds and runs numerical parity tests for the C++ backend. The CuPy algorithm contract
  is tested without requiring a GPU; actual CUDA performance must be validated on the
  target NVIDIA system.
