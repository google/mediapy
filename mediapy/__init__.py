# Copyright 2024 The mediapy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`mediapy`: Read/write/show images and videos in an IPython/Jupyter notebook.

[**[GitHub source]**](https://github.com/google/mediapy) &nbsp;
[**[API docs]**](https://google.github.io/mediapy/) &nbsp;
[**[PyPI package]**](https://pypi.org/project/mediapy/) &nbsp;
[**[Colab
example]**](https://colab.research.google.com/github/google/mediapy/blob/main/mediapy_examples.ipynb)

See the [example
notebook](https://github.com/google/mediapy/blob/main/mediapy_examples.ipynb),
or better yet, [**open it in
Colab**](https://colab.research.google.com/github/google/mediapy/blob/main/mediapy_examples.ipynb).

## Image examples

Display an image (2D or 3D `numpy` array):
```python
checkerboard = np.kron([[0, 1] * 16, [1, 0] * 16] * 16, np.ones((4, 4)))
show_image(checkerboard)
```

Read and display an image (either local or from the Web):
```python
IMAGE = 'https://github.com/hhoppe/data/raw/main/image.png'
show_image(read_image(IMAGE))
```

Read and display an image from a local file:
```python
!wget -q -O /tmp/burano.png {IMAGE}
show_image(read_image('/tmp/burano.png'))
```

Show titled images side-by-side:
```python
images = {
    'original': checkerboard,
    'darkened': checkerboard * 0.7,
    'random': np.random.rand(32, 32, 3),
}
show_images(images, vmin=0.0, vmax=1.0, border=True, height=64)
```

Compare two images using an interactive slider:
```python
compare_images([checkerboard, np.random.rand(128, 128, 3)])
```

## Video examples

Display a video (an iterable of images, e.g., a 3D or 4D array):
```python
video = moving_circle((100, 100), num_images=10)
show_video(video, fps=10)
```

Show the video frames side-by-side:
```python
show_images(video, columns=6, border=True, height=64)
```

Show the frames with their indices:
```python
show_images({f'{i}': image for i, image in enumerate(video)}, width=32)
```

Read and display a video (either local or from the Web):
```python
VIDEO = 'https://github.com/hhoppe/data/raw/main/video.mp4'
show_video(read_video(VIDEO))
```

Create and display a looping two-frame GIF video:
```python
image1 = resize_image(np.random.rand(10, 10, 3), (50, 50))
show_video([image1, image1 * 0.8], fps=2, codec='gif')
```

Darken a video frame-by-frame:
```python
output_path = '/tmp/out.mp4'
with VideoReader(VIDEO) as r:
  darken_image = lambda image: to_float01(image) * 0.5
  with VideoWriter(output_path, shape=r.shape, fps=r.fps, bps=r.bps) as w:
    for image in r:
      w.add_image(darken_image(image))
```
"""

from __future__ import annotations

__docformat__ = 'google'
__version__ = '1.2.2'
__version_info__ = tuple(int(num) for num in __version__.split('.'))

import base64
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import contextlib
import functools
import importlib
import io
import itertools
import math
import numbers
import os  # Package only needed for typing.TYPE_CHECKING.
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import typing
from typing import Any
import urllib.request

import IPython.display
import matplotlib.pyplot
import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageOps


if not hasattr(PIL.Image, 'Resampling'):  # Allow Pillow<9.0.
  PIL.Image.Resampling = PIL.Image  # type: ignore

# Selected and reordered here for pdoc documentation.
__all__ = [
    'show_image',
    'show_images',
    'compare_images',
    'show_video',
    'show_videos',
    'read_image',
    'write_image',
    'read_video',
    'write_video',
    'VideoReader',
    'VideoWriter',
    'VideoMetadata',
    'compress_image',
    'decompress_image',
    'compress_video',
    'decompress_video',
    'html_from_compressed_image',
    'html_from_compressed_video',
    'resize_image',
    'resize_video',
    'to_rgb',
    'to_type',
    'to_float01',
    'to_uint8',
    'set_output_height',
    'set_max_output_height',
    'color_ramp',
    'moving_circle',
    'set_show_save_dir',
    'set_ffmpeg',
    'video_is_available',
]

if typing.TYPE_CHECKING:
  _ArrayLike = npt.ArrayLike
  _DTypeLike = npt.DTypeLike
  _NDArray = np.ndarray[Any, Any]
  _DType = np.dtype[Any]
else:
  # Create named types for use in the `pdoc` documentation.
  _ArrayLike = typing.TypeVar('_ArrayLike')
  _DTypeLike = typing.TypeVar('_DTypeLike')
  _NDArray = typing.TypeVar('_NDArray')
  _DType = typing.TypeVar('_DType')  # pylint: disable=invalid-name

_IPYTHON_HTML_SIZE_LIMIT = 20_000_000
_T = typing.TypeVar('_T')
_Path = typing.Union[str, 'os.PathLike[str]']

_IMAGE_COMPARISON_HTML = """\
<script
  defer
  src="https://unpkg.com/img-comparison-slider@7/dist/index.js"
></script>
<link
  rel="stylesheet"
  href="https://unpkg.com/img-comparison-slider@7/dist/styles.css"
/>

<img-comparison-slider>
  <img slot="first" src="data:image/png;base64,{b64_1}" />
  <img slot="second" src="data:image/png;base64,{b64_2}" />
</img-comparison-slider>
"""

# ** Miscellaneous.


class _Config:
  ffmpeg_name_or_path: _Path = 'ffmpeg'
  show_save_dir: _Path | None = None


_config = _Config()


def _open(path: _Path, *args: Any, **kwargs: Any) -> Any:
  """Opens the file; this is a hook for the built-in `open()`."""
  return open(path, *args, **kwargs)


def _path_is_local(path: _Path) -> bool:
  """Returns True if the path is in the filesystem accessible by `ffmpeg`."""
  del path
  return True


def _search_for_ffmpeg_path() -> str | None:
  """Returns a path to the ffmpeg program, or None if not found."""
  if filename := shutil.which(_config.ffmpeg_name_or_path):
    return str(filename)
  return None


def _print_err(*args: str, **kwargs: Any) -> None:
  """Prints arguments to stderr immediately."""
  kwargs = {**dict(file=sys.stderr, flush=True), **kwargs}
  print(*args, **kwargs)


def _chunked(
    iterable: Iterable[_T], n: int | None = None
) -> Iterator[tuple[_T, ...]]:
  """Returns elements collected as tuples of length at most `n` if not None."""

  def take(n: int, iterable: Iterable[_T]) -> tuple[_T, ...]:
    return tuple(itertools.islice(iterable, n))

  return iter(functools.partial(take, n, iter(iterable)), ())


def _peek_first(iterator: Iterable[_T]) -> tuple[_T, Iterable[_T]]:
  """Given an iterator, returns first element and re-initialized iterator.

  >>> first_image, images = _peek_first(moving_circle())

  Args:
    iterator: An input iterator or iterable.

  Returns:
    A tuple (first_element, iterator_reinitialized) containing:
      first_element: The first element of the input.
      iterator_reinitialized: A clone of the original iterator/iterable.
  """
  # Inspired from https://stackoverflow.com/a/12059829/1190077
  peeker, iterator_reinitialized = itertools.tee(iterator)
  first = next(peeker)
  return first, iterator_reinitialized


def _check_2d_shape(shape: tuple[int, int]) -> None:
  """Checks that `shape` is of the form (height, width) with two integers."""
  if len(shape) != 2:
    raise ValueError(f'Shape {shape} is not of the form (height, width).')
  if not all(isinstance(i, numbers.Integral) for i in shape):
    raise ValueError(f'Shape {shape} contains non-integers.')


def _run(args: str | Sequence[str]) -> None:
  """Executes command, printing output from stdout and stderr.

  Args:
    args: Command to execute, which can be either a string or a sequence of word
      strings, as in `subprocess.run()`.  If `args` is a string, the shell is
      invoked to interpret it.

  Raises:
    RuntimeError: If the command's exit code is nonzero.
  """
  proc = subprocess.run(
      args,
      shell=isinstance(args, str),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
      universal_newlines=True,
  )
  print(proc.stdout, end='', flush=True)
  if proc.returncode:
    raise RuntimeError(
        f"Command '{proc.args}' failed with code {proc.returncode}."
    )


def _display_html(text: str, /) -> None:
  """In a Jupyter notebook, display the HTML `text`."""
  IPython.display.display(IPython.display.HTML(text))  # type: ignore


def set_ffmpeg(name_or_path: _Path) -> None:
  """Specifies the name or path for the `ffmpeg` external program.

  The `ffmpeg` program is required for compressing and decompressing video.
  (It is used in `read_video`, `write_video`, `show_video`, `show_videos`,
  etc.)

  Args:
    name_or_path: Either a filename within a directory of `os.environ['PATH']`
      or a filepath.  The default setting is 'ffmpeg'.
  """
  _config.ffmpeg_name_or_path = name_or_path


def set_output_height(num_pixels: int) -> None:
  """Overrides the height of the current output cell, if using Colab."""
  try:
    # We want to fail gracefully for non-Colab IPython notebooks.
    output = importlib.import_module('google.colab.output')
    s = f'google.colab.output.setIframeHeight("{num_pixels}px")'
    output.eval_js(s)
  except (ModuleNotFoundError, AttributeError):
    pass


def set_max_output_height(num_pixels: int) -> None:
  """Sets the maximum height of the current output cell, if using Colab."""
  try:
    # We want to fail gracefully for non-Colab IPython notebooks.
    output = importlib.import_module('google.colab.output')
    s = (
        'google.colab.output.setIframeHeight('
        f'0, true, {{maxHeight: {num_pixels}}})'
    )
    output.eval_js(s)
  except (ModuleNotFoundError, AttributeError):
    pass


# ** Type conversions.


def _as_valid_media_type(dtype: _DTypeLike) -> _DType:
  """Returns validated media data type."""
  dtype = np.dtype(dtype)
  if not issubclass(dtype.type, (np.unsignedinteger, np.floating)):
    raise ValueError(
        f'Type {dtype} is not a valid media data type (uint or float).'
    )
  return dtype


def _as_valid_media_array(x: _ArrayLike) -> _NDArray:
  """Converts to ndarray (if not already), and checks validity of data type."""
  a = np.asarray(x)
  if a.dtype == bool:
    a = a.astype(np.uint8) * np.iinfo(np.uint8).max
  _as_valid_media_type(a.dtype)
  return a


def to_type(array: _ArrayLike, dtype: _DTypeLike) -> _NDArray:
  """Returns media array converted to specified type.

  A "media array" is one in which the dtype is either a floating-point type
  (np.float32 or np.float64) or an unsigned integer type.  The array values are
  assumed to lie in the range [0.0, 1.0] for floating-point values, and in the
  full range for unsigned integers, e.g. [0, 255] for np.uint8.

  Conversion between integers and floats maps uint(0) to 0.0 and uint(MAX) to
  1.0.  The input array may also be of type bool, whereby True maps to
  uint(MAX) or 1.0.  The values are scaled and clamped as appropriate during
  type conversions.

  Args:
    array: Input array-like object (floating-point, unsigned int, or bool).
    dtype: Desired output type (floating-point or unsigned int).

  Returns:
    Array `a` if it is already of the specified dtype, else a converted array.
  """
  a = np.asarray(array)
  dtype = np.dtype(dtype)
  del array
  if a.dtype != bool:
    _as_valid_media_type(a.dtype)  # Verify that 'a' has a valid dtype.
  if a.dtype == bool:
    result = a.astype(dtype)
    if np.issubdtype(dtype, np.unsignedinteger):
      result = result * dtype.type(np.iinfo(dtype).max)
  elif a.dtype == dtype:
    result = a
  elif np.issubdtype(dtype, np.unsignedinteger):
    if np.issubdtype(a.dtype, np.unsignedinteger):
      src_max: float = np.iinfo(a.dtype).max
    else:
      a = np.clip(a, 0.0, 1.0)
      src_max = 1.0
    dst_max = np.iinfo(dtype).max
    if dst_max <= np.iinfo(np.uint16).max:
      scale = np.array(dst_max / src_max, dtype=np.float32)
      result = (a * scale + 0.5).astype(dtype)
    elif dst_max <= np.iinfo(np.uint32).max:
      result = (a.astype(np.float64) * (dst_max / src_max) + 0.5).astype(dtype)
    else:
      # https://stackoverflow.com/a/66306123/
      a = a.astype(np.float64) * (dst_max / src_max) + 0.5
      dst = np.atleast_1d(a)
      values_too_large = dst >= np.float64(dst_max)
      with np.errstate(invalid='ignore'):
        dst = dst.astype(dtype)
      dst[values_too_large] = dst_max
      result = dst if a.ndim > 0 else dst[0]
  else:
    assert np.issubdtype(dtype, np.floating)
    result = a.astype(dtype)
    if np.issubdtype(a.dtype, np.unsignedinteger):
      result = result / dtype.type(np.iinfo(a.dtype).max)
  return result


def to_float01(a: _ArrayLike, dtype: _DTypeLike = np.float32) -> _NDArray:
  """If array has unsigned integers, rescales them to the range [0.0, 1.0].

  Scaling is such that uint(0) maps to 0.0 and uint(MAX) maps to 1.0.  See
  `to_type`.

  Args:
    a: Input array.
    dtype: Desired floating-point type if rescaling occurs.

  Returns:
    A new array of dtype values in the range [0.0, 1.0] if the input array `a`
    contains unsigned integers; otherwise, array `a` is returned unchanged.
  """
  a = np.asarray(a)
  dtype = np.dtype(dtype)
  if not np.issubdtype(dtype, np.floating):
    raise ValueError(f'Type {dtype} is not floating-point.')
  if np.issubdtype(a.dtype, np.floating):
    return a
  return to_type(a, dtype)


def to_uint8(a: _ArrayLike) -> _NDArray:
  """Returns array converted to uint8 values; see `to_type`."""
  return to_type(a, np.uint8)


# ** Functions to generate example image and video data.


def color_ramp(
    shape: tuple[int, int] = (64, 64), *, dtype: _DTypeLike = np.float32
) -> _NDArray:
  """Returns an image of a red-green color gradient.

  This is useful for quick experimentation and testing.  See also
  `moving_circle` to generate a sample video.

  Args:
    shape: 2D spatial dimensions (height, width) of generated image.
    dtype: Type (uint or floating) of resulting pixel values.
  """
  _check_2d_shape(shape)
  dtype = _as_valid_media_type(dtype)
  yx = (np.moveaxis(np.indices(shape), 0, -1) + 0.5) / shape
  image = np.insert(yx, 2, 0.0, axis=-1)
  return to_type(image, dtype)


def moving_circle(
    shape: tuple[int, int] = (256, 256),
    num_images: int = 10,
    *,
    dtype: _DTypeLike = np.float32,
) -> _NDArray:
  """Returns a video of a circle moving in front of a color ramp.

  This is useful for quick experimentation and testing.  See also `color_ramp`
  to generate a sample image.

  >>> show_video(moving_circle((480, 640), 60), fps=60)

  Args:
    shape: 2D spatial dimensions (height, width) of generated video.
    num_images: Number of video frames.
    dtype: Type (uint or floating) of resulting pixel values.
  """
  _check_2d_shape(shape)
  dtype = np.dtype(dtype)

  def generate_image(image_index: int) -> _NDArray:
    """Returns a video frame image."""
    image = color_ramp(shape, dtype=dtype)
    yx = np.moveaxis(np.indices(shape), 0, -1)
    center = shape[0] * 0.6, shape[1] * (image_index + 0.5) / num_images
    radius_squared = (min(shape) * 0.1) ** 2
    inside = np.sum((yx - center) ** 2, axis=-1) < radius_squared
    white_circle_color = 1.0, 1.0, 1.0
    if np.issubdtype(dtype, np.unsignedinteger):
      white_circle_color = to_type([white_circle_color], dtype)[0]
    image[inside] = white_circle_color
    return image

  return np.array([generate_image(i) for i in range(num_images)])


# ** Color-space conversions.

# Same matrix values as in two sources:
# https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L377
# https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/ops/image_ops_impl.py#L2754
_YUV_FROM_RGB_MATRIX = np.array(
    [
        [0.299, -0.14714119, 0.61497538],
        [0.587, -0.28886916, -0.51496512],
        [0.114, 0.43601035, -0.10001026],
    ],
    dtype=np.float32,
)
_RGB_FROM_YUV_MATRIX = np.linalg.inv(_YUV_FROM_RGB_MATRIX)
_YUV_CHROMA_OFFSET = np.array([0.0, 0.5, 0.5], dtype=np.float32)


def yuv_from_rgb(rgb: _ArrayLike) -> _NDArray:
  """Returns the RGB image/video mapped to YUV [0,1] color space.

  Note that the "YUV" color space used by video compressors is actually YCbCr!

  Args:
    rgb: Input image in sRGB space.
  """
  rgb = to_float01(rgb)
  if rgb.shape[-1] != 3:
    raise ValueError(f'The last dimension in {rgb.shape} is not 3.')
  return rgb @ _YUV_FROM_RGB_MATRIX + _YUV_CHROMA_OFFSET


def rgb_from_yuv(yuv: _ArrayLike) -> _NDArray:
  """Returns the YUV image/video mapped to RGB [0,1] color space."""
  yuv = to_float01(yuv)
  if yuv.shape[-1] != 3:
    raise ValueError(f'The last dimension in {yuv.shape} is not 3.')
  return (yuv - _YUV_CHROMA_OFFSET) @ _RGB_FROM_YUV_MATRIX


# Same matrix values as in
# https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L1654
# and https://en.wikipedia.org/wiki/YUV#Studio_swing_for_BT.601
_YCBCR_FROM_RGB_MATRIX = np.array(
    [
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.0],
        [112.0, -93.786, -18.214],
    ],
    dtype=np.float32,
).transpose()
_RGB_FROM_YCBCR_MATRIX = np.linalg.inv(_YCBCR_FROM_RGB_MATRIX)
_YCBCR_OFFSET = np.array([16.0, 128.0, 128.0], dtype=np.float32)
# Note that _YCBCR_FROM_RGB_MATRIX =~ _YUV_FROM_RGB_MATRIX * [219, 256, 182];
# https://en.wikipedia.org/wiki/YUV: "Y' values are conventionally shifted and
# scaled to the range [16, 235] (referred to as studio swing or 'TV levels')";
# "studio range of 16-240 for U and V".  (Where does value 182 come from?)


def ycbcr_from_rgb(rgb: _ArrayLike) -> _NDArray:
  """Returns the RGB image/video mapped to YCbCr [0,1] color space.

  The YCbCr color space is the one called "YUV" by video compressors.

  Args:
    rgb: Input image in sRGB space.
  """
  rgb = to_float01(rgb)
  if rgb.shape[-1] != 3:
    raise ValueError(f'The last dimension in {rgb.shape} is not 3.')
  return (rgb @ _YCBCR_FROM_RGB_MATRIX + _YCBCR_OFFSET) / 255.0


def rgb_from_ycbcr(ycbcr: _ArrayLike) -> _NDArray:
  """Returns the YCbCr image/video mapped to RGB [0,1] color space."""
  ycbcr = to_float01(ycbcr)
  if ycbcr.shape[-1] != 3:
    raise ValueError(f'The last dimension in {ycbcr.shape} is not 3.')
  return (ycbcr * 255.0 - _YCBCR_OFFSET) @ _RGB_FROM_YCBCR_MATRIX


# ** Image processing.


def _pil_image(image: _ArrayLike, mode: str | None = None) -> PIL.Image.Image:
  """Returns a PIL image given a numpy matrix (either uint8 or float [0,1])."""
  image = _as_valid_media_array(image)
  if image.ndim not in (2, 3):
    raise ValueError(f'Image shape {image.shape} is neither 2D nor 3D.')
  pil_image: PIL.Image.Image = PIL.Image.fromarray(image, mode=mode)  # type: ignore[no-untyped-call]
  return pil_image


def resize_image(image: _ArrayLike, shape: tuple[int, int]) -> _NDArray:
  """Resizes image to specified spatial dimensions using a Lanczos filter.

  Args:
    image: Array-like 2D or 3D object, where dtype is uint or floating-point.
    shape: 2D spatial dimensions (height, width) of output image.

  Returns:
    A resampled image whose spatial dimensions match `shape`.
  """
  image = _as_valid_media_array(image)
  if image.ndim not in (2, 3):
    raise ValueError(f'Image shape {image.shape} is neither 2D nor 3D.')
  _check_2d_shape(shape)

  # A PIL image can be multichannel only if it has 3 or 4 uint8 channels,
  # and it can be resized only if it is uint8 or float32.
  supported_single_channel = (
      np.issubdtype(image.dtype, np.floating) or image.dtype == np.uint8
  ) and image.ndim == 2
  supported_multichannel = (
      image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] in (3, 4)
  )
  if supported_single_channel or supported_multichannel:
    return np.array(
        _pil_image(image).resize(
            shape[::-1], resample=PIL.Image.Resampling.LANCZOS
        ),
        dtype=image.dtype,
    )
  if image.ndim == 2:
    # We convert to floating-point for resizing and convert back.
    return to_type(resize_image(to_float01(image), shape), image.dtype)
  # We resize each image channel individually.
  return np.dstack(
      [resize_image(channel, shape) for channel in np.moveaxis(image, -1, 0)]
  )


# ** Video processing.


def resize_video(video: Iterable[_NDArray], shape: tuple[int, int]) -> _NDArray:
  """Resizes `video` to specified spatial dimensions using a Lanczos filter.

  Args:
    video: Iterable of images.
    shape: 2D spatial dimensions (height, width) of output video.

  Returns:
    A resampled video whose spatial dimensions match `shape`.
  """
  _check_2d_shape(shape)
  return np.array([resize_image(image, shape) for image in video])


# ** General I/O.


def _is_url(path_or_url: _Path) -> bool:
  return isinstance(path_or_url, str) and path_or_url.startswith(
      ('http://', 'https://', 'file://')
  )


def read_contents(path_or_url: _Path) -> bytes:
  """Returns the contents of the file specified by either a path or URL."""
  data: bytes
  if _is_url(path_or_url):
    assert isinstance(path_or_url, str)
    with urllib.request.urlopen(path_or_url) as response:
      data = response.read()
  else:
    with _open(path_or_url, 'rb') as f:
      data = f.read()
  return data


@contextlib.contextmanager
def _read_via_local_file(path_or_url: _Path) -> Iterator[str]:
  """Context to copy a remote file locally to read from it.

  Args:
    path_or_url: File, which may be remote.

  Yields:
    The name of a local file which may be a copy of a remote file.
  """
  if _is_url(path_or_url) or not _path_is_local(path_or_url):
    suffix = pathlib.Path(path_or_url).suffix
    with tempfile.TemporaryDirectory() as directory_name:
      tmp_path = pathlib.Path(directory_name) / f'file{suffix}'
      tmp_path.write_bytes(read_contents(path_or_url))
      yield str(tmp_path)
  else:
    yield str(path_or_url)


@contextlib.contextmanager
def _write_via_local_file(path: _Path) -> Iterator[str]:
  """Context to write a temporary local file and subsequently copy it remotely.

  Args:
    path: File, which may be remote.

  Yields:
    The name of a local file which may be subsequently copied remotely.
  """
  if _path_is_local(path):
    yield str(path)
  else:
    suffix = pathlib.Path(path).suffix
    with tempfile.TemporaryDirectory() as directory_name:
      tmp_path = pathlib.Path(directory_name) / f'file{suffix}'
      yield str(tmp_path)
      with _open(path, mode='wb') as f:
        f.write(tmp_path.read_bytes())


class set_show_save_dir:  # pylint: disable=invalid-name
  """Save all titled output from `show_*()` calls into files.

  If the specified `directory` is not None, all titled images and videos
  displayed by `show_image`, `show_images`, `show_video`, and `show_videos` are
  also saved as files within the directory.

  It can be used either to set the state or as a context manager:

  >>> set_show_save_dir('/tmp')
  >>> show_image(color_ramp(), title='image1')  # Creates /tmp/image1.png.
  >>> show_video(moving_circle(), title='video2')  # Creates /tmp/video2.mp4.
  >>> set_show_save_dir(None)

  >>> with set_show_save_dir('/tmp'):
  ...   show_image(color_ramp(), title='image1')  # Creates /tmp/image1.png.
  ...   show_video(moving_circle(), title='video2')  # Creates /tmp/video2.mp4.
  """

  def __init__(self, directory: _Path | None):
    self._old_show_save_dir = _config.show_save_dir
    _config.show_save_dir = directory

  def __enter__(self) -> None:
    pass

  def __exit__(self, *_: Any) -> None:
    _config.show_save_dir = self._old_show_save_dir


# ** Image I/O.


def read_image(
    path_or_url: _Path,
    *,
    apply_exif_transpose: bool = True,
    dtype: _DTypeLike = None,
) -> _NDArray:
  """Returns an image read from a file path or URL.

  Decoding is performed using `PIL`, which supports `uint8` images with 1, 3,
  or 4 channels and `uint16` images with a single channel.

  Args:
    path_or_url: Path of input file.
    apply_exif_transpose: If True, rotate image according to EXIF orientation.
    dtype: Data type of the returned array.  If None, `np.uint8` or `np.uint16`
      is inferred automatically.
  """
  data = read_contents(path_or_url)
  return decompress_image(data, dtype, apply_exif_transpose)


def write_image(
    path: _Path, image: _ArrayLike, fmt: str = 'png', **kwargs: Any
) -> None:
  """Writes an image to a file.

  Encoding is performed using `PIL`, which supports `uint8` images with 1, 3,
  or 4 channels and `uint16` images with a single channel.

  File format is explicitly provided by `fmt` and not inferred by `path`.

  Args:
    path: Path of output file.
    image: Array-like object.  If its type is float, it is converted to np.uint8
      using `to_uint8` (thus clamping to the input to the range [0.0, 1.0]).
      Otherwise it must be np.uint8 or np.uint16.
    fmt: Desired compression encoding, e.g. 'png'.
    **kwargs: Additional parameters for `PIL.Image.save()`.
  """
  image = _as_valid_media_array(image)
  if np.issubdtype(image.dtype, np.floating):
    image = to_uint8(image)
  with _open(path, 'wb') as f:
    _pil_image(image).save(f, format=fmt, **kwargs)


def to_rgb(
    array: _ArrayLike,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Callable[[_ArrayLike], _NDArray] = 'gray',
) -> _NDArray:
  """Maps scalar values to RGB using value bounds and a color map.

  Args:
    array: Scalar values, with arbitrary shape.
    vmin: Explicit min value for remapping; if None, it is obtained as the
      minimum finite value of `array`.
    vmax: Explicit max value for remapping; if None, it is obtained as the
      maximum finite value of `array`.
    cmap: A `pyplot` color map or callable, to map from 1D value to 3D or 4D
      color.

  Returns:
    A new array in which each element is affinely mapped from [vmin, vmax]
    to [0.0, 1.0] and then color-mapped.
  """
  a = _as_valid_media_array(array)
  del array
  # For future numpy version 1.7.0:
  # vmin = np.amin(a, where=np.isfinite(a)) if vmin is None else vmin
  # vmax = np.amax(a, where=np.isfinite(a)) if vmax is None else vmax
  vmin = np.amin(np.where(np.isfinite(a), a, np.inf)) if vmin is None else vmin
  vmax = np.amax(np.where(np.isfinite(a), a, -np.inf)) if vmax is None else vmax
  a = (a.astype('float') - vmin) / (vmax - vmin + np.finfo(float).eps)
  if isinstance(cmap, str):
    if hasattr(matplotlib, 'colormaps'):
      rgb_from_scalar: Any = matplotlib.colormaps[cmap]  # Newer version.
    else:
      rgb_from_scalar = matplotlib.pyplot.cm.get_cmap(cmap)  # type: ignore # pylint: disable=no-member
  else:
    rgb_from_scalar = cmap
  a = rgb_from_scalar(a)
  # If there is a fully opaque alpha channel, remove it.
  if a.shape[-1] == 4 and np.all(to_float01(a[..., 3])) == 1.0:
    a = a[..., :3]
  return a


def compress_image(
    image: _ArrayLike, *, fmt: str = 'png', **kwargs: Any
) -> bytes:
  """Returns a buffer containing a compressed image.

  Args:
    image: Array in a format supported by `PIL`, e.g. np.uint8 or np.uint16.
    fmt: Desired compression encoding, e.g. 'png'.
    **kwargs: Options for `PIL.save()`, e.g. `optimize=True` for greater
      compression.
  """
  image = _as_valid_media_array(image)
  with io.BytesIO() as output:
    _pil_image(image).save(output, format=fmt, **kwargs)
    return output.getvalue()


def decompress_image(
    data: bytes, dtype: _DTypeLike = None, apply_exif_transpose: bool = True
) -> _NDArray:
  """Returns an image from a compressed data buffer.

  Decoding is performed using `PIL`, which supports `uint8` images with 1, 3,
  or 4 channels and `uint16` images with a single channel.

  Args:
    data: Buffer containing compressed image.
    dtype: Data type of the returned array.  If None, `np.uint8` or `np.uint16`
      is inferred automatically.
    apply_exif_transpose: If True, rotate image according to EXIF orientation.
  """
  pil_image = PIL.Image.open(io.BytesIO(data))
  if apply_exif_transpose:
    tmp_image = PIL.ImageOps.exif_transpose(pil_image)  # Future: in_place=True.
    assert tmp_image
    pil_image = tmp_image
  if dtype is None:
    dtype = np.uint16 if pil_image.mode.startswith('I') else np.uint8
  return np.array(pil_image, dtype=dtype)


def html_from_compressed_image(
    data: bytes,
    width: int,
    height: int,
    *,
    title: str | None = None,
    border: bool | str = False,
    pixelated: bool = True,
    fmt: str = 'png',
) -> str:
  """Returns an HTML string with an image tag containing encoded data.

  Args:
    data: Compressed image bytes.
    width: Width of HTML image in pixels.
    height: Height of HTML image in pixels.
    title: Optional text shown centered above image.
    border: If `bool`, whether to place a black boundary around the image, or if
      `str`, the boundary CSS style.
    pixelated: If True, sets the CSS style to 'image-rendering: pixelated;'.
    fmt: Compression encoding.
  """
  b64 = base64.b64encode(data).decode('utf-8')
  if isinstance(border, str):
    border = f'{border}; '
  elif border:
    border = 'border:1px solid black; '
  else:
    border = ''
  s_pixelated = 'pixelated' if pixelated else 'auto'
  s = (
      f'<img width="{width}" height="{height}"'
      f' style="{border}image-rendering:{s_pixelated}; object-fit:cover;"'
      f' src="data:image/{fmt};base64,{b64}"/>'
  )
  if title is not None:
    s = f"""<div style="display:flex; align-items:left;">
      <div style="display:flex; flex-direction:column; align-items:center;">
      <div>{title}</div><div>{s}</div></div></div>"""
  return s


def _get_width_height(
    width: int | None, height: int | None, shape: tuple[int, int]
) -> tuple[int, int]:
  """Returns (width, height) given optional parameters and image shape."""
  assert len(shape) == 2, shape
  if width and height:
    return width, height
  if width and not height:
    return width, int(width * (shape[0] / shape[1]) + 0.5)
  if height and not width:
    return int(height * (shape[1] / shape[0]) + 0.5), height
  return shape[::-1]


def _ensure_mapped_to_rgb(
    image: _ArrayLike,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Callable[[_ArrayLike], _NDArray] = 'gray',
) -> _NDArray:
  """Ensure image is mapped to RGB."""
  image = _as_valid_media_array(image)
  if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 3, 4))):
    raise ValueError(
        f'Image with shape {image.shape} is neither a 2D array'
        ' nor a 3D array with 1, 3, or 4 channels.'
    )
  if image.ndim == 3 and image.shape[2] == 1:
    image = image[:, :, 0]
  if image.ndim == 2:
    image = to_rgb(image, vmin=vmin, vmax=vmax, cmap=cmap)
  return image


def show_image(
    image: _ArrayLike, *, title: str | None = None, **kwargs: Any
) -> str | None:
  """Displays an image in the notebook and optionally saves it to a file.

  See `show_images`.

  >>> show_image(np.random.rand(100, 100))
  >>> show_image(np.random.randint(0, 256, size=(80, 80, 3), dtype='uint8'))
  >>> show_image(np.random.rand(10, 10) - 0.5, cmap='bwr', height=100)
  >>> show_image(read_image('/tmp/image.png'))
  >>> url = 'https://github.com/hhoppe/data/raw/main/image.png'
  >>> show_image(read_image(url))

  Args:
    image: 2D array-like, or 3D array-like with 1, 3, or 4 channels.
    title: Optional text shown centered above the image.
    **kwargs: See `show_images`.

  Returns:
    html string if `return_html` is `True`.
  """
  return show_images([np.asarray(image)], [title], **kwargs)


def show_images(
    images: Iterable[_ArrayLike] | Mapping[str, _ArrayLike],
    titles: Iterable[str | None] | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    downsample: bool = True,
    columns: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Callable[[_ArrayLike], _NDArray] = 'gray',
    border: bool | str = False,
    ylabel: str = '',
    html_class: str = 'show_images',
    pixelated: bool | None = None,
    return_html: bool = False,
) -> str | None:
  """Displays a row of images in the IPython/Jupyter notebook.

  If a directory has been specified using `set_show_save_dir`, also saves each
  titled image to a file in that directory based on its title.

  >>> image1, image2 = np.random.rand(64, 64, 3), color_ramp((64, 64))
  >>> show_images([image1, image2])
  >>> show_images({'random image': image1, 'color ramp': image2}, height=128)
  >>> show_images([image1, image2] * 5, columns=4, border=True)

  Args:
    images: Iterable of images, or dictionary of `{title: image}`.  Each image
      must be either a 2D array or a 3D array with 1, 3, or 4 channels.
    titles: Optional strings shown above the corresponding images.
    width: Optional, overrides displayed width (in pixels).
    height: Optional, overrides displayed height (in pixels).
    downsample: If True, each image whose width or height is greater than the
      specified `width` or `height` is resampled to the display resolution. This
      improves antialiasing and reduces the size of the notebook.
    columns: Optional, maximum number of images per row.
    vmin: For single-channel image, explicit min value for display.
    vmax: For single-channel image, explicit max value for display.
    cmap: For single-channel image, `pyplot` color map or callable to map 1D to
      3D color.
    border: If `bool`, whether to place a black boundary around the image, or if
      `str`, the boundary CSS style.
    ylabel: Text (rotated by 90 degrees) shown on the left of each row.
    html_class: CSS class name used in definition of HTML element.
    pixelated: If True, sets the CSS style to 'image-rendering: pixelated;'; if
      False, sets 'image-rendering: auto'; if None, uses pixelated rendering
      only on images for which `width` or `height` introduces magnification.
    return_html: If `True` return the raw HTML `str` instead of displaying.

  Returns:
    html string if `return_html` is `True`.
  """
  if isinstance(images, Mapping):
    if titles is not None:
      raise ValueError('Cannot have images dictionary and titles parameter.')
    list_titles, list_images = list(images.keys()), list(images.values())
  else:
    list_images = list(images)
    list_titles = [None] * len(list_images) if titles is None else list(titles)
    if len(list_images) != len(list_titles):
      raise ValueError(
          'Number of images does not match number of titles'
          f' ({len(list_images)} vs {len(list_titles)}).'
      )

  list_images = [
      _ensure_mapped_to_rgb(image, vmin=vmin, vmax=vmax, cmap=cmap)
      for image in list_images
  ]

  def maybe_downsample(image: _NDArray) -> _NDArray:
    shape: tuple[int, int] = image.shape[:2]  # type: ignore[assignment]
    w, h = _get_width_height(width, height, shape)
    if w < shape[1] or h < shape[0]:
      image = resize_image(image, (h, w))
    return image

  if downsample:
    list_images = [maybe_downsample(image) for image in list_images]
  png_datas = [compress_image(to_uint8(image)) for image in list_images]

  for title, png_data in zip(list_titles, png_datas):
    if title is not None and _config.show_save_dir:
      path = pathlib.Path(_config.show_save_dir) / f'{title}.png'
      with _open(path, mode='wb') as f:
        f.write(png_data)

  def html_from_compressed_images() -> str:
    html_strings = []
    for image, title, png_data in zip(list_images, list_titles, png_datas):
      w, h = _get_width_height(width, height, image.shape[:2])
      magnified = h > image.shape[0] or w > image.shape[1]
      pixelated2 = pixelated if pixelated is not None else magnified
      html_strings.append(
          html_from_compressed_image(
              png_data, w, h, title=title, border=border, pixelated=pixelated2
          )
      )
    # Create single-row tables each with no more than 'columns' elements.
    table_strings = []
    for row_html_strings in _chunked(html_strings, columns):
      td = '<td style="padding:1px;">'
      s = ''.join(f'{td}{e}</td>' for e in row_html_strings)
      if ylabel:
        style = 'writing-mode:vertical-lr; transform:rotate(180deg);'
        s = f'{td}<span style="{style}">{ylabel}</span></td>' + s
      table_strings.append(
          f'<table class="{html_class}"'
          f' style="border-spacing:0px;"><tr>{s}</tr></table>'
      )
    return ''.join(table_strings)

  s = html_from_compressed_images()
  while len(s) > _IPYTHON_HTML_SIZE_LIMIT * 0.5:
    list_images = [image[::2, ::2] for image in list_images]
    png_datas = [compress_image(to_uint8(image)) for image in list_images]
    s = html_from_compressed_images()
  if return_html:
    return s
  _display_html(s)
  return None


def compare_images(
    images: Iterable[_ArrayLike],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Callable[[_ArrayLike], _NDArray] = 'gray',
) -> None:
  """Compare two images using an interactive slider.

  Displays an HTML slider component to interactively swipe between two images.
  The slider functionality requires that the web browser have Internet access.
  See additional info in `https://github.com/sneas/img-comparison-slider`.

  >>> image1, image2 = np.random.rand(64, 64, 3), color_ramp((64, 64))
  >>> compare_images([image1, image2])

  Args:
    images: Iterable of images.  Each image must be either a 2D array or a 3D
      array with 1, 3, or 4 channels.  There must be exactly two images.
    vmin: For single-channel image, explicit min value for display.
    vmax: For single-channel image, explicit max value for display.
    cmap: For single-channel image, `pyplot` color map or callable to map 1D to
      3D color.
  """
  list_images = [
      _ensure_mapped_to_rgb(image, vmin=vmin, vmax=vmax, cmap=cmap)
      for image in images
  ]
  if len(list_images) != 2:
    raise ValueError('The number of images must be 2.')
  png_datas = [compress_image(to_uint8(image)) for image in list_images]
  b64_1, b64_2 = [
      base64.b64encode(png_data).decode('utf-8') for png_data in png_datas
  ]
  s = _IMAGE_COMPARISON_HTML.replace('{b64_1}', b64_1).replace('{b64_2}', b64_2)
  _display_html(s)


# ** Video I/O.


def _filename_suffix_from_codec(codec: str) -> str:
  return '.gif' if codec == 'gif' else '.mp4'


def _get_ffmpeg_path() -> str:
  path = _search_for_ffmpeg_path()
  if not path:
    raise RuntimeError(
        f"Program '{_config.ffmpeg_name_or_path}' is not found;"
        " perhaps install ffmpeg using 'apt install ffmpeg'."
    )
  return path


def video_is_available() -> bool:
  """Returns True if the program `ffmpeg` is found.

  See also `set_ffmpeg`.
  """
  return _search_for_ffmpeg_path() is not None


class VideoMetadata(typing.NamedTuple):
  """Represents the data stored in a video container header.

  Attributes:
    num_images: Number of frames that is expected from the video stream.  This
      is estimated from the framerate and the duration stored in the video
      header, so it might be inexact.  We set the value to -1 if number of
      frames is not found in the header.
    shape: The dimensions (height, width) of each video frame.
    fps: The framerate in frames per second.
    bps: The estimated bitrate of the video stream in bits per second, retrieved
      from the video header.
  """

  num_images: int
  shape: tuple[int, int]
  fps: float
  bps: int | None


def _get_video_metadata(path: _Path) -> VideoMetadata:
  """Returns attributes of video stored in the specified local file."""
  if not pathlib.Path(path).is_file():
    raise RuntimeError(f"Video file '{path}' is not found.")
  command = [
      _get_ffmpeg_path(),
      '-nostdin',
      '-i',
      str(path),
      '-acodec',
      'copy',
      '-vcodec',
      'copy',
      '-f',
      'null',
      '-',
  ]
  with subprocess.Popen(
      command, stderr=subprocess.PIPE, encoding='utf-8'
  ) as proc:
    _, err = proc.communicate()
  bps = fps = num_images = width = height = rotation = None
  for line in err.split('\n'):
    if match := re.search(r', bitrate: *([\d.]+) kb/s', line):
      bps = int(match.group(1)) * 1000
    if matches := re.findall(r'frame= *(\d+) ', line):
      num_images = int(matches[-1])
    if 'Stream #0:' in line and ': Video:' in line:
      if not (match := re.search(r', (\d+)x(\d+)', line)):
        raise RuntimeError(f'Unable to parse video dimensions in line {line}')
      width, height = int(match.group(1)), int(match.group(2))
      if match := re.search(r', ([\d.]+) fps', line):
        fps = float(match.group(1))
      elif str(path).endswith('.gif'):
        # Some GIF files lack a framerate attribute; use a reasonable default.
        fps = 10
      else:
        raise RuntimeError(f'Unable to parse video framerate in line {line}')
    if match := re.fullmatch(r'\s*rotate\s*:\s*(\d+)', line):
      rotation = int(match.group(1))
    if match := re.fullmatch(r'.*rotation of (-?\d+).*\sdegrees', line):
      rotation = int(match.group(1))
  if not num_images:
    num_images = -1
  if not width:
    raise RuntimeError(f'Unable to parse video header: {err}')
  # By default, ffmpeg enables "-autorotate"; we just fix the dimensions.
  if rotation in (90, 270, -90, -270):
    width, height = height, width
  assert height is not None and width is not None
  shape = height, width
  assert fps is not None
  return VideoMetadata(num_images, shape, fps, bps)


class _VideoIO:
  """Base class for `VideoReader` and `VideoWriter`."""

  def _get_pix_fmt(self, dtype: _DType, image_format: str) -> str:
    """Returns ffmpeg pix_fmt given data type and image format."""
    native_endian_suffix = {'little': 'le', 'big': 'be'}[sys.byteorder]
    return {
        np.uint8: {
            'rgb': 'rgb24',
            'yuv': 'yuv444p',
            'gray': 'gray',
        },
        np.uint16: {
            'rgb': 'rgb48' + native_endian_suffix,
            'yuv': 'yuv444p16' + native_endian_suffix,
            'gray': 'gray16' + native_endian_suffix,
        },
    }[dtype.type][image_format]


class VideoReader(_VideoIO):
  """Context to read a compressed video as an iterable over its images.

  >>> with VideoReader('/tmp/river.mp4') as reader:
  ...   print(f'Video has {reader.num_images} images with shape={reader.shape},'
  ...         f' at {reader.fps} frames/sec and {reader.bps} bits/sec.')
  ...   for image in reader:
  ...     print(image.shape)

  >>> with VideoReader('/tmp/river.mp4') as reader:
  ...   video = np.array(tuple(reader))

  >>> url = 'https://github.com/hhoppe/data/raw/main/video.mp4'
  >>> with VideoReader(url) as reader:
  ...   show_video(reader)

  Attributes:
    path_or_url: Location of input video.
    output_format: Format of output images (default 'rgb').  If 'rgb', each
      image has shape=(height, width, 3) with R, G, B values.  If 'yuv', each
      image has shape=(height, width, 3) with Y, U, V values.  If 'gray', each
      image has shape=(height, width).
    dtype: Data type for output images.  The default is `np.uint8`.  Use of
      `np.uint16` allows reading 10-bit or 12-bit data without precision loss.
    metadata: Object storing the information retrieved from the video header.
      Its attributes are copied as attributes in this class.
    num_images: Number of frames that is expected from the video stream.  This
      is estimated from the framerate and the duration stored in the video
      header, so it might be inexact.
    shape: The dimensions (height, width) of each video frame.
    fps: The framerate in frames per second.
    bps: The estimated bitrate of the video stream in bits per second, retrieved
      from the video header.
  """

  path_or_url: _Path
  output_format: str
  dtype: _DType
  metadata: VideoMetadata
  num_images: int
  shape: tuple[int, int]
  fps: float
  bps: int | None
  _num_bytes_per_image: int

  def __init__(
      self,
      path_or_url: _Path,
      *,
      output_format: str = 'rgb',
      dtype: _DTypeLike = np.uint8,
  ):
    if output_format not in {'rgb', 'yuv', 'gray'}:
      raise ValueError(
          f'Output format {output_format} is not rgb, yuv, or gray.'
      )
    self.path_or_url = path_or_url
    self.output_format = output_format
    self.dtype = np.dtype(dtype)
    if self.dtype.type not in (np.uint8, np.uint16):
      raise ValueError(f'Type {dtype} is not np.uint8 or np.uint16.')
    self._read_via_local_file: Any = None
    self._popen: subprocess.Popen[bytes] | None = None
    self._proc: subprocess.Popen[bytes] | None = None

  def __enter__(self) -> 'VideoReader':
    ffmpeg_path = _get_ffmpeg_path()
    try:
      self._read_via_local_file = _read_via_local_file(self.path_or_url)
      # pylint: disable-next=no-member
      tmp_name = self._read_via_local_file.__enter__()

      self.metadata = _get_video_metadata(tmp_name)
      self.num_images, self.shape, self.fps, self.bps = self.metadata
      pix_fmt = self._get_pix_fmt(self.dtype, self.output_format)
      num_channels = {'rgb': 3, 'yuv': 3, 'gray': 1}[self.output_format]
      bytes_per_channel = self.dtype.itemsize
      self._num_bytes_per_image = (
          math.prod(self.shape) * num_channels * bytes_per_channel
      )

      command = [
          ffmpeg_path,
          '-v',
          'panic',
          '-nostdin',
          '-i',
          tmp_name,
          '-vcodec',
          'rawvideo',
          '-f',
          'image2pipe',
          '-pix_fmt',
          pix_fmt,
          '-vsync',
          'vfr',
          '-',
      ]
      self._popen = subprocess.Popen(
          command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
      )
      self._proc = self._popen.__enter__()
    except Exception:
      self.__exit__(None, None, None)
      raise
    return self

  def __exit__(self, *_: Any) -> None:
    self.close()

  def read(self) -> _NDArray | None:
    """Reads a video image frame (or None if at end of file).

    Returns:
      A numpy array in the format specified by `output_format`, i.e., a 3D
      array with 3 color channels, except for format 'gray' which is 2D.
    """
    assert self._proc, 'Error: reading from an already closed context.'
    stdout = self._proc.stdout
    assert stdout is not None
    data = stdout.read(self._num_bytes_per_image)
    if not data:  # Due to either end-of-file or subprocess error.
      self.close()  # Raises exception if subprocess had error.
      return None  # To indicate end-of-file.
    assert len(data) == self._num_bytes_per_image
    image = np.frombuffer(data, dtype=self.dtype)
    if self.output_format == 'rgb':
      image = image.reshape(*self.shape, 3)
    elif self.output_format == 'yuv':  # Convert from planar YUV to pixel YUV.
      image = np.moveaxis(image.reshape(3, *self.shape), 0, 2)
    elif self.output_format == 'gray':  # Generate 2D rather than 3D ndimage.
      image = image.reshape(*self.shape)
    else:
      raise AssertionError
    return image

  def __iter__(self) -> Iterator[_NDArray]:
    while True:
      image = self.read()
      if image is None:
        return
      yield image

  def close(self) -> None:
    """Terminates video reader.  (Called automatically at end of context.)"""
    if self._popen:
      self._popen.__exit__(None, None, None)
      self._popen = None
      self._proc = None
    if self._read_via_local_file:
      # pylint: disable-next=no-member
      self._read_via_local_file.__exit__(None, None, None)
      self._read_via_local_file = None


class VideoWriter(_VideoIO):
  """Context to write a compressed video.

  >>> shape = 480, 640
  >>> with VideoWriter('/tmp/v.mp4', shape, fps=60) as writer:
  ...   for image in moving_circle(shape, num_images=60):
  ...     writer.add_image(image)
  >>> show_video(read_video('/tmp/v.mp4'))


  Bitrate control may be specified using at most one of: `bps`, `qp`, or `crf`.
  If none are specified, `qp` is set to a default value.
  See https://slhck.info/video/2017/03/01/rate-control.html

  If codec is 'gif', the args `bps`, `qp`, `crf`, and `encoded_format` are
  ignored.

  Attributes:
    path: Output video.  Its suffix (e.g. '.mp4') determines the video container
      format.  The suffix must be '.gif' if the codec is 'gif'.
    shape: 2D spatial dimensions (height, width) of video image frames.  The
      dimensions must be even if 'encoded_format' has subsampled chroma (e.g.,
      'yuv420p' or 'yuv420p10le').
    codec: Compression algorithm as defined by "ffmpeg -codecs" (e.g., 'h264',
      'hevc', 'vp9', or 'gif').
    metadata: Optional VideoMetadata object whose `fps` and `bps` attributes are
      used if not specified as explicit parameters.
    fps: Frames-per-second framerate (default is 60.0 except 25.0 for 'gif').
    bps: Requested average bits-per-second bitrate (default None).
    qp: Quantization parameter for video compression quality (default None).
    crf: Constant rate factor for video compression quality (default None).
    ffmpeg_args: Additional arguments for `ffmpeg` command, e.g. '-g 30' to
      introduce I-frames, or '-bf 0' to omit B-frames.
    input_format: Format of input images (default 'rgb').  If 'rgb', each image
      has shape=(height, width, 3) or (height, width).  If 'yuv', each image has
      shape=(height, width, 3) with Y, U, V values.  If 'gray', each image has
      shape=(height, width).
    dtype: Expected data type for input images (any float input images are
      converted to `dtype`).  The default is `np.uint8`.  Use of `np.uint16` is
      necessary when encoding >8 bits/channel.
    encoded_format: Pixel format as defined by `ffmpeg -pix_fmts`, e.g.,
      'yuv420p' (2x2-subsampled chroma), 'yuv444p' (full-res chroma),
      'yuv420p10le' (10-bit per channel), etc.  The default (None) selects
      'yuv420p' if all shape dimensions are even, else 'yuv444p'.
  """

  def __init__(
      self,
      path: _Path,
      shape: tuple[int, int],
      *,
      codec: str = 'h264',
      metadata: VideoMetadata | None = None,
      fps: float | None = None,
      bps: int | None = None,
      qp: int | None = None,
      crf: float | None = None,
      ffmpeg_args: str | Sequence[str] = '',
      input_format: str = 'rgb',
      dtype: _DTypeLike = np.uint8,
      encoded_format: str | None = None,
  ) -> None:
    _check_2d_shape(shape)
    if fps is None and metadata:
      fps = metadata.fps
    if fps is None:
      fps = 25.0 if codec == 'gif' else 60.0
    if fps <= 0.0:
      raise ValueError(f'Frame-per-second value {fps} is invalid.')
    if bps is None and metadata:
      bps = metadata.bps
    bps = int(bps) if bps is not None else None
    if bps is not None and bps <= 0:
      raise ValueError(f'Bitrate value {bps} is invalid.')
    if qp is not None and (not isinstance(qp, int) or qp <= 0):
      raise ValueError(
          f'Quantization parameter {qp} is not a positive integer.'
      )
    num_rate_specifications = sum(x is not None for x in (bps, qp, crf))
    if num_rate_specifications > 1:
      raise ValueError(
          f'Must specify at most one of bps, qp, or crf ({bps}, {qp}, {crf}).'
      )
    ffmpeg_args = (
        shlex.split(ffmpeg_args)
        if isinstance(ffmpeg_args, str)
        else list(ffmpeg_args)
    )
    if input_format not in {'rgb', 'yuv', 'gray'}:
      raise ValueError(f'Input format {input_format} is not rgb, yuv, or gray.')
    dtype = np.dtype(dtype)
    if dtype.type not in (np.uint8, np.uint16):
      raise ValueError(f'Type {dtype} is not np.uint8 or np.uint16.')
    self.path = pathlib.Path(path)
    self.shape = shape
    all_dimensions_are_even = all(dim % 2 == 0 for dim in shape)
    if encoded_format is None:
      encoded_format = 'yuv420p' if all_dimensions_are_even else 'yuv444p'
    if not all_dimensions_are_even and encoded_format.startswith(
        ('yuv42', 'yuvj42')
    ):
      raise ValueError(
          f'With encoded_format {encoded_format}, video dimensions must be'
          f' even, but shape is {shape}.'
      )
    self.fps = fps
    self.codec = codec
    self.bps = bps
    self.qp = qp
    self.crf = crf
    self.ffmpeg_args = ffmpeg_args
    self.input_format = input_format
    self.dtype = dtype
    self.encoded_format = encoded_format
    if num_rate_specifications == 0 and not ffmpeg_args:
      qp = 20 if math.prod(self.shape) <= 640 * 480 else 28
    self._bitrate_args = (
        (['-vb', f'{bps}'] if bps is not None else [])
        + (['-qp', f'{qp}'] if qp is not None else [])
        + (['-vb', '0', '-crf', f'{crf}'] if crf is not None else [])
    )
    if self.codec == 'gif':
      if self.path.suffix != '.gif':
        raise ValueError(f"File '{self.path}' does not have a .gif suffix.")
      self.encoded_format = 'pal8'
      self._bitrate_args = []
      video_filter = 'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse'
      # Less common (and likely less useful) is a per-frame color palette:
      # video_filter = ('split[s0][s1];[s0]palettegen=stats_mode=single[p];'
      #                 '[s1][p]paletteuse=new=1')
      self.ffmpeg_args = ['-vf', video_filter, '-f', 'gif'] + self.ffmpeg_args
    self._write_via_local_file: Any = None
    self._popen: subprocess.Popen[bytes] | None = None
    self._proc: subprocess.Popen[bytes] | None = None

  def __enter__(self) -> 'VideoWriter':
    ffmpeg_path = _get_ffmpeg_path()
    input_pix_fmt = self._get_pix_fmt(self.dtype, self.input_format)
    try:
      self._write_via_local_file = _write_via_local_file(self.path)
      # pylint: disable-next=no-member
      tmp_name = self._write_via_local_file.__enter__()

      # Writing to stdout using ('-f', 'mp4', '-') would require
      # ('-movflags', 'frag_keyframe+empty_moov') and the result is nonportable.
      height, width = self.shape
      command = (
          [
              ffmpeg_path,
              '-v',
              'error',
              '-f',
              'rawvideo',
              '-vcodec',
              'rawvideo',
              '-pix_fmt',
              input_pix_fmt,
              '-s',
              f'{width}x{height}',
              '-r',
              f'{self.fps}',
              '-i',
              '-',
              '-an',
              '-vcodec',
              self.codec,
              '-pix_fmt',
              self.encoded_format,
          ]
          + self._bitrate_args
          + self.ffmpeg_args
          + ['-y', tmp_name]
      )
      self._popen = subprocess.Popen(
          command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
      )
      self._proc = self._popen.__enter__()
    except Exception:
      self.__exit__(None, None, None)
      raise
    return self

  def __exit__(self, *_: Any) -> None:
    self.close()

  def add_image(self, image: _NDArray) -> None:
    """Writes a video frame.

    Args:
      image: Array whose dtype and first two dimensions must match the `dtype`
        and `shape` specified in `VideoWriter` initialization.  If
        `input_format` is 'gray', the image must be 2D.  For the 'rgb'
        input_format, the image may be either 2D (interpreted as grayscale) or
        3D with three (R, G, B) channels.  For the 'yuv' input_format, the image
        must be 3D with three (Y, U, V) channels.

    Raises:
      RuntimeError: If there is an error writing to the output file.
    """
    assert self._proc, 'Error: writing to an already closed context.'
    if issubclass(image.dtype.type, (np.floating, np.bool_)):
      image = to_type(image, self.dtype)
    if image.dtype != self.dtype:
      raise ValueError(f'Image type {image.dtype} != {self.dtype}.')
    if self.input_format == 'gray':
      if image.ndim != 2:
        raise ValueError(f'Image dimensions {image.shape} are not 2D.')
    else:
      if image.ndim == 2 and self.input_format == 'rgb':
        image = np.dstack((image, image, image))
      if not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError(f'Image dimensions {image.shape} are invalid.')
    if image.shape[:2] != self.shape:
      raise ValueError(
          f'Image dimensions {image.shape[:2]} do not match'
          f' those of the initialized video {self.shape}.'
      )
    if self.input_format == 'yuv':  # Convert from per-pixel YUV to planar YUV.
      image = np.moveaxis(image, 2, 0)
    data = image.tobytes()
    stdin = self._proc.stdin
    assert stdin is not None
    if stdin.write(data) != len(data):
      self._proc.wait()
      stderr = self._proc.stderr
      assert stderr is not None
      s = stderr.read().decode()
      raise RuntimeError(f"Error writing '{self.path}': {s}")

  def close(self) -> None:
    """Finishes writing the video.  (Called automatically at end of context.)"""
    if self._popen:
      assert self._proc, 'Error: closing an already closed context.'
      stdin = self._proc.stdin
      assert stdin is not None
      stdin.close()
      if self._proc.wait():
        stderr = self._proc.stderr
        assert stderr is not None
        s = stderr.read().decode()
        raise RuntimeError(f"Error writing '{self.path}': {s}")
      self._popen.__exit__(None, None, None)
      self._popen = None
      self._proc = None
    if self._write_via_local_file:
      # pylint: disable-next=no-member
      self._write_via_local_file.__exit__(None, None, None)
      self._write_via_local_file = None


class _VideoArray(npt.NDArray[Any]):
  """Wrapper to add a VideoMetadata `metadata` attribute to a numpy array."""

  metadata: VideoMetadata | None

  def __new__(
      cls: typing.Type['_VideoArray'],
      input_array: _NDArray,
      metadata: VideoMetadata | None = None,
  ) -> '_VideoArray':
    obj: _VideoArray = np.asarray(input_array).view(cls)
    obj.metadata = metadata
    return obj

  def __array_finalize__(self, obj: Any) -> None:
    if obj is None:
      return
    self.metadata = getattr(obj, 'metadata', None)


def read_video(path_or_url: _Path, **kwargs: Any) -> _VideoArray:
  """Returns an array containing all images read from a compressed video file.

  >>> video = read_video('/tmp/river.mp4')
  >>> print(f'The framerate is {video.metadata.fps} frames/s.')
  >>> show_video(video)

  >>> url = 'https://github.com/hhoppe/data/raw/main/video.mp4'
  >>> show_video(read_video(url))

  Args:
    path_or_url: Input video file.
    **kwargs: Additional parameters for `VideoReader`.

  Returns:
    A 4D `numpy` array with dimensions (frame, height, width, channel), or a 3D
    array if `output_format` is specified as 'gray'.  The returned array has an
    attribute `metadata` containing `VideoMetadata` information.  This enables
    `show_video` to retrieve the framerate in `metadata.fps`.  Note that the
    metadata attribute is lost in most subsequent `numpy` operations.
  """
  with VideoReader(path_or_url, **kwargs) as reader:
    return _VideoArray(np.array(tuple(reader)), metadata=reader.metadata)


def write_video(path: _Path, images: Iterable[_NDArray], **kwargs: Any) -> None:
  """Writes images to a compressed video file.

  >>> video = moving_circle((480, 640), num_images=60)
  >>> write_video('/tmp/v.mp4', video, fps=60, qp=18)
  >>> show_video(read_video('/tmp/v.mp4'))

  Args:
    path: Output video file.
    images: Iterable over video frames, e.g. a 4D array or a list of 2D or 3D
      arrays.
    **kwargs: Additional parameters for `VideoWriter`.
  """
  first_image, images = _peek_first(images)
  shape: tuple[int, int] = first_image.shape[:2]  # type: ignore[assignment]
  dtype = first_image.dtype
  if dtype == bool:
    dtype = np.dtype(np.uint8)
  elif np.issubdtype(dtype, np.floating):
    dtype = np.dtype(np.uint16)
  kwargs = {'metadata': getattr(images, 'metadata', None), **kwargs}
  with VideoWriter(path, shape=shape, dtype=dtype, **kwargs) as writer:
    for image in images:
      writer.add_image(image)


def compress_video(
    images: Iterable[_NDArray], *, codec: str = 'h264', **kwargs: Any
) -> bytes:
  """Returns a buffer containing a compressed video.

  The video container is 'mp4' except when `codec` is 'gif'.

  >>> video = read_video('/tmp/river.mp4')
  >>> data = compress_video(video, bps=10_000_000)
  >>> print(len(data))

  >>> data = compress_video(moving_circle((100, 100), num_images=10), fps=10)

  Args:
    images: Iterable over video frames.
    codec: Compression algorithm as defined by `ffmpeg -codecs` (e.g., 'h264',
      'hevc', 'vp9', or 'gif').
    **kwargs: Additional parameters for `VideoWriter`.

  Returns:
    A bytes buffer containing the compressed video.
  """
  suffix = _filename_suffix_from_codec(codec)
  with tempfile.TemporaryDirectory() as directory_name:
    tmp_path = pathlib.Path(directory_name) / f'file{suffix}'
    write_video(tmp_path, images, codec=codec, **kwargs)
    return tmp_path.read_bytes()


def decompress_video(data: bytes, **kwargs: Any) -> _NDArray:
  """Returns video images from an MP4-compressed data buffer."""
  with tempfile.TemporaryDirectory() as directory_name:
    tmp_path = pathlib.Path(directory_name) / 'file.mp4'
    tmp_path.write_bytes(data)
    return read_video(tmp_path, **kwargs)


def html_from_compressed_video(
    data: bytes,
    width: int,
    height: int,
    *,
    title: str | None = None,
    border: bool | str = False,
    loop: bool = True,
    autoplay: bool = True,
) -> str:
  """Returns an HTML string with a video tag containing H264-encoded data.

  Args:
    data: MP4-compressed video bytes.
    width: Width of HTML video in pixels.
    height: Height of HTML video in pixels.
    title: Optional text shown centered above the video.
    border: If `bool`, whether to place a black boundary around the image, or if
      `str`, the boundary CSS style.
    loop: If True, the playback repeats forever.
    autoplay: If True, video playback starts without having to click.
  """
  b64 = base64.b64encode(data).decode('utf-8')
  if isinstance(border, str):
    border = f'{border}; '
  elif border:
    border = 'border:1px solid black; '
  else:
    border = ''
  options = (
      f'controls width="{width}" height="{height}"'
      f' style="{border}object-fit:cover;"'
      f'{" loop" if loop else ""}'
      f'{" autoplay muted" if autoplay else ""}'
  )
  s = f"""<video {options}>
      <source src="data:video/mp4;base64,{b64}" type="video/mp4"/>
      This browser does not support the video tag.
      </video>"""
  if title is not None:
    s = f"""<div style="display:flex; align-items:left;">
      <div style="display:flex; flex-direction:column; align-items:center;">
      <div>{title}</div><div>{s}</div></div></div>"""
  return s


def show_video(
    images: Iterable[_NDArray], *, title: str | None = None, **kwargs: Any
) -> str | None:
  """Displays a video in the IPython notebook and optionally saves it to a file.

  See `show_videos`.

  >>> video = read_video('https://github.com/hhoppe/data/raw/main/video.mp4')
  >>> show_video(video, title='River video')

  >>> show_video(moving_circle((80, 80), num_images=10), fps=5, border=True)

  >>> show_video(read_video('/tmp/river.mp4'))

  Args:
    images: Iterable of video frames (e.g., a 4D array or a list of 2D or 3D
      arrays).
    title: Optional text shown centered above the video.
    **kwargs: See `show_videos`.

  Returns:
    html string if `return_html` is `True`.
  """
  return show_videos([images], [title], **kwargs)


def show_videos(
    videos: Iterable[Iterable[_NDArray]] | Mapping[str, Iterable[_NDArray]],
    titles: Iterable[str | None] | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    downsample: bool = True,
    columns: int | None = None,
    fps: float | None = None,
    bps: int | None = None,
    qp: int | None = None,
    codec: str = 'h264',
    ylabel: str = '',
    html_class: str = 'show_videos',
    return_html: bool = False,
    **kwargs: Any,
) -> str | None:
  """Displays a row of videos in the IPython notebook.

  Creates HTML with `<video>` tags containing embedded H264-encoded bytestrings.
  If `codec` is set to 'gif', we instead use `<img>` tags containing embedded
  GIF-encoded bytestrings.  Note that the resulting GIF animations skip frames
  when the `fps` period is not a multiple of 10 ms units (GIF frame delay
  units).  Encoding at `fps` = 20.0, 25.0, or 50.0 works fine.

  If a directory has been specified using `set_show_save_dir`, also saves each
  titled video to a file in that directory based on its title.

  Args:
    videos: Iterable of videos, or dictionary of `{title: video}`.  Each video
      must be an iterable of images.  If a video object has a `metadata`
      (`VideoMetadata`) attribute, its `fps` field provides a default framerate.
    titles: Optional strings shown above the corresponding videos.
    width: Optional, overrides displayed width (in pixels).
    height: Optional, overrides displayed height (in pixels).
    downsample: If True, each video whose width or height is greater than the
      specified `width` or `height` is resampled to the display resolution. This
      improves antialiasing and reduces the size of the notebook.
    columns: Optional, maximum number of videos per row.
    fps: Frames-per-second framerate (default is 60.0 except 25.0 for GIF).
    bps: Bits-per-second bitrate (default None).
    qp: Quantization parameter for video compression quality (default None).
    codec: Compression algorithm; must be either 'h264' or 'gif'.
    ylabel: Text (rotated by 90 degrees) shown on the left of each row.
    html_class: CSS class name used in definition of HTML element.
    return_html: If `True` return the raw HTML `str` instead of displaying.
    **kwargs: Additional parameters (`border`, `loop`, `autoplay`) for
      `html_from_compressed_video`.

  Returns:
    html string if `return_html` is `True`.
  """
  if isinstance(videos, Mapping):
    if titles is not None:
      raise ValueError(
          'Cannot have both a video dictionary and a titles parameter.'
      )
    list_titles = list(videos.keys())
    list_videos = list(videos.values())
  else:
    list_videos = list(typing.cast('Iterable[_NDArray]', videos))
    list_titles = [None] * len(list_videos) if titles is None else list(titles)
    if len(list_videos) != len(list_titles):
      raise ValueError(
          'Number of videos does not match number of titles'
          f' ({len(list_videos)} vs {len(list_titles)}).'
      )
  if codec not in {'h264', 'gif'}:
    raise ValueError(f'Codec {codec} is neither h264 or gif.')

  html_strings = []
  for video, title in zip(list_videos, list_titles):
    metadata: VideoMetadata | None = getattr(video, 'metadata', None)
    first_image, video = _peek_first(video)
    w, h = _get_width_height(width, height, first_image.shape[:2])
    if downsample and (w < first_image.shape[1] or h < first_image.shape[0]):
      # Not resize_video() because each image may have different depth and type.
      video = [resize_image(image, (h, w)) for image in video]
      first_image = video[0]
    data = compress_video(
        video, metadata=metadata, fps=fps, bps=bps, qp=qp, codec=codec
    )
    if title is not None and _config.show_save_dir:
      suffix = _filename_suffix_from_codec(codec)
      path = pathlib.Path(_config.show_save_dir) / f'{title}{suffix}'
      with _open(path, mode='wb') as f:
        f.write(data)
    if codec == 'gif':
      pixelated = h > first_image.shape[0] or w > first_image.shape[1]
      html_string = html_from_compressed_image(
          data, w, h, title=title, fmt='gif', pixelated=pixelated, **kwargs
      )
    else:
      html_string = html_from_compressed_video(
          data, w, h, title=title, **kwargs
      )
    html_strings.append(html_string)

  # Create single-row tables each with no more than 'columns' elements.
  table_strings = []
  for row_html_strings in _chunked(html_strings, columns):
    td = '<td style="padding:1px;">'
    s = ''.join(f'{td}{e}</td>' for e in row_html_strings)
    if ylabel:
      style = 'writing-mode:vertical-lr; transform:rotate(180deg);'
      s = f'{td}<span style="{style}">{ylabel}</span></td>' + s
    table_strings.append(
        f'<table class="{html_class}"'
        f' style="border-spacing:0px;"><tr>{s}</tr></table>'
    )
  s = ''.join(table_strings)
  if return_html:
    return s
  _display_html(s)
  return None


# Local Variables:
# fill-column: 80
# End:
