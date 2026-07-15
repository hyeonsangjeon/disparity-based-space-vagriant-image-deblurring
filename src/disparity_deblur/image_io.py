from pathlib import Path

import cv2
import numpy as np


RASTER_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def read_input_image(
    path: str | Path,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Input image does not exist: {path}. "
            "Use an actual image file path."
        )
    if path.suffix.lower() == ".raw":
        if height is None or width is None:
            raise ValueError("RAW input requires both --height and --width")
        return read_rgb24(path, height, width)
    if path.suffix.lower() not in RASTER_EXTENSIONS:
        supported = ", ".join(sorted(RASTER_EXTENSIONS | {".raw"}))
        raise ValueError(f"unsupported input extension {path.suffix!r}; use {supported}")
    if (height is None) != (width is None):
        raise ValueError("--height and --width must be supplied together")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode image: {path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    actual_height, actual_width = image.shape[:2]
    if height is not None and (actual_height != height or actual_width != width):
        raise ValueError(
            f"{path} is {actual_width}x{actual_height}; "
            f"requested {width}x{height}"
        )
    return image.astype(np.float32) / 255.0


def read_rgb24(path: str | Path, height: int, width: int) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"RAW input does not exist: {path}. "
            "Use an actual image file path."
        )
    expected = height * width * 3
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != expected:
        raise ValueError(
            f"{path} contains {data.size} bytes; expected {expected} "
            f"for {width}x{height} RGB24"
        )
    return data.reshape(height, width, 3).astype(np.float32) / 255.0


def write_rgb24(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_uint8(image).tofile(path)


def write_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _to_uint8(image)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write {path}")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image.shape}")
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
