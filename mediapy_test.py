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

"""Tests for package mediapy."""

import io
import pathlib
import re
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import IPython
import matplotlib
import mediapy as media
import numpy as np

# pylint: disable=missing-function-docstring, protected-access
# pylint: disable=too-many-public-methods

_TEST_TYPES = ['uint8', 'uint16', 'uint32', 'float32', 'float64']
_TEST_SHAPES1 = [(13, 21, 3), (14, 38, 2), (16, 21, 1), (18, 20), (17, 19)]
_TEST_SHAPES2 = [
    (128, 128, 3),
    (128, 160, 1),
    (160, 128),
    (64, 64, 3),
    (64, 64),
]
_TEST_SHAPES3 = [(), (1,), (2, 2)]


def _rms_diff(a, b) -> float:
  """Compute the root-mean-square of the difference between two arrays."""
  a = np.array(a, dtype=np.float64)
  b = np.array(b, dtype=np.float64)
  if a.shape != b.shape:
    raise ValueError(f'Shapes {a.shape} and {b.shape} do not match.')
  return np.sqrt(np.mean(np.square(a - b)))


class MediapyTest(parameterized.TestCase):
  """Tests for mediapy package."""

  def assert_all_equal(self, a, b):
    if not np.all(np.asarray(a) == np.asarray(b)):
      self.fail(f'{a} and {b} differ.')

  def assert_all_close(self, a, b, **kwargs):
    if not np.allclose(a, b, **kwargs):
      self.fail(f'{a} and {b} are not close enough.')

  def _check_similar(self, original_array, new_array, max_rms, msg=None):
    """Verifies that the rms error between two arrays is less than max_rms."""
    self.assert_all_equal(original_array.shape, new_array.shape)
    rms = _rms_diff(new_array, original_array)
    self.assertLess(rms, max_rms, msg)

  def test_chunked(self):
    self.assertEqual(list(media._chunked(range(0), 3)), [])
    self.assertEqual(list(media._chunked(range(1), 3)), [(0,)])
    self.assertEqual(list(media._chunked(range(2), 3)), [(0, 1)])
    self.assertEqual(list(media._chunked(range(3), 3)), [(0, 1, 2)])
    self.assertEqual(list(media._chunked(range(4), 3)), [(0, 1, 2), (3,)])
    self.assertEqual(list(media._chunked(range(5), 3)), [(0, 1, 2), (3, 4)])

    self.assertEqual(list(media._chunked(range(0), 1)), [])
    self.assertEqual(list(media._chunked(range(1), 1)), [(0,)])
    self.assertEqual(list(media._chunked(range(2), 1)), [(0,), (1,)])
    self.assertEqual(list(media._chunked(range(3), 1)), [(0,), (1,), (2,)])

    self.assertEqual(list(media._chunked(range(0), None)), [])
    self.assertEqual(list(media._chunked(range(1), None)), [(0,)])
    self.assertEqual(list(media._chunked(range(2), None)), [(0, 1)])
    self.assertEqual(list(media._chunked(range(3), None)), [(0, 1, 2)])

  def test_peek_first_on_generator(self):
    generator = range(1, 5)
    first, generator = media._peek_first(generator)
    self.assertEqual(first, 1)
    self.assert_all_equal(tuple(generator), [1, 2, 3, 4])

  def test_peek_first_on_container(self):
    container = [1, 2, 3, 4]
    first, container = media._peek_first(container)
    self.assertEqual(first, 1)
    self.assert_all_equal(tuple(container), [1, 2, 3, 4])

  def test_run_string(self):
    with mock.patch('sys.stdout', io.StringIO()) as mock_stdout:
      media._run('echo ab cd')
      self.assertEqual(mock_stdout.getvalue(), 'ab cd\n')
    if sys.platform == 'linux':
      with mock.patch('sys.stdout', io.StringIO()) as mock_stdout:
        media._run('echo "$((17 + 22))"')
        self.assertEqual(mock_stdout.getvalue(), '39\n')
      with mock.patch('sys.stdout', io.StringIO()) as mock_stdout:
        media._run('/bin/bash -c "echo $((17 + 22))"')
        self.assertEqual(mock_stdout.getvalue(), '39\n')
      with self.assertRaisesRegex(RuntimeError, 'failed with code 3'):
        media._run('exit 3')

  def test_run_args_sequence(self):
    with mock.patch('sys.stdout', io.StringIO()) as mock_stdout:
      media._run(['echo', 'ef', 'gh'])
      self.assertEqual(mock_stdout.getvalue(), 'ef gh\n')
    if sys.platform == 'linux':
      with mock.patch('sys.stdout', io.StringIO()) as mock_stdout:
        media._run(['/bin/bash', '-c', 'echo $((17 + 22))'])
        self.assertEqual(mock_stdout.getvalue(), '39\n')

  def test_to_type(self):
    def check(src, dtype, expected):
      output = media.to_type(src, dtype)
      self.assertEqual(output.dtype.type, np.dtype(dtype).type)
      self.assert_all_equal(output, expected)

    max32 = 4_294_967_295
    b = np.array([False, True, False])
    self.assertEqual(b.dtype, bool)
    check(b, np.uint8, [0, 255, 0])
    check(b, np.uint16, [0, 65535, 0])
    check(b, np.uint32, [0, max32, 0])
    check(b, np.float32, [0.0, 1.0, 0.0])
    check(b, np.float64, [0.0, 1.0, 0.0])

    u8 = np.array([3, 255], dtype=np.uint8)
    check(u8, 'uint8', [3, 255])
    check(u8, 'uint16', [int(3 / 255 * 65535 + 0.5), 65535])
    check(u8, 'uint32', [int(3 / 255 * max32 + 0.5), max32])
    check(u8, 'float32', [np.float32(3 / 255), 1.0])
    check(u8, 'float64', [3 / 255, 1.0])

    u16 = np.array([57, 65535], dtype=np.uint16)
    check(u16, np.uint8, [0, 255])
    check(u16, np.uint16, [57, 65535])
    check(u16, np.uint32, [int(57 / 65535 * max32 + 0.5), max32])
    check(u16, np.float32, [np.float32(57 / 65535), 1.0])
    check(u16, 'float', [57 / 65535, 1.0])

    u32 = np.array([100_000, max32], dtype=np.uint32)
    check(u32, 'uint8', [0, 255])
    check(u32, 'uint16', [2, 65535])
    check(u32, 'uint32', u32)
    check(u32, 'float32', [np.float32(100_000 / max32), 1.0])
    check(u32, 'float64', [100_000 / max32, 1.0])

    f32 = np.array([0.0, 0.4, 1.0], dtype=np.float32)
    check(f32, np.uint8, [0, int(np.float32(0.4) * 255 + 0.5), 255])
    check(f32, np.uint16, [0, int(np.float32(0.4) * 65535 + 0.5), 65535])
    check(f32, np.uint32, [0, int(np.float32(0.4) * max32 + 0.5), max32])
    check(f32, np.float32, [0.0, np.float32(0.4), 1.0])
    check(f32, np.float64, [0.0, np.float32(0.4), 1.0])

    f64 = np.array([0.0, 0.4, 1.0], dtype=np.float64)
    check(f64, np.uint8, [0, int(0.4 * 255 + 0.5), 255])
    check(f64, np.uint16, [0, int(0.4 * 65535 + 0.5), 65535])
    check(f64, np.uint32, [0, int(0.4 * max32 + 0.5), max32])
    check(f64, np.float32, [0.0, np.float32(0.4), 1.0])
    check(f64, np.float64, [0.0, 0.4, 1.0])

    # An array with data type 'uint64' is possible, but it is awkward to process
    # exactly because it requires more than float64 intermediate precision.

  @parameterized.parameters(
      zip(_TEST_TYPES + ['uint64', 'bool'], _TEST_TYPES, _TEST_SHAPES3)
  )
  def test_to_type_extreme_value(self, src_dtype, dst_dtype, shape):
    max_of_type = dict(
        bool=True,
        uint8=255,
        uint16=65535,
        uint32=4294967295,
        uint64=18446744073709551615,
        float32=1.0,
        float64=1.0,
    )
    src_value = max_of_type[src_dtype]
    src = np.full(shape, src_value, dtype=src_dtype)
    dst = media.to_type(src, dst_dtype)
    dst_value = dst.flat[0]
    expected_value = max_of_type[dst_dtype]
    msg = f'{src} {dst}'
    self.assertEqual(dst.dtype, dst_dtype, msg=msg)
    self.assertEqual(dst.shape, src.shape, msg=msg)
    self.assertEqual(dst_value, expected_value, msg=msg)

  def test_to_float01(self):
    self.assert_all_close(
        media.to_float01(np.array([0, 1, 128, 254, 255], dtype=np.uint8)),
        [0 / 255, 1 / 255, 128 / 255, 254 / 255, 255 / 255],
    )
    self.assert_all_close(
        media.to_float01(np.array([0, 1, 128, 254, 65535], dtype=np.uint16)),
        [0 / 65535, 1 / 65535, 128 / 65535, 254 / 65535, 65535 / 65535],
    )
    a = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    self.assertIs(media.to_float01(a), a)
    a = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32)
    self.assertIs(media.to_float01(a), a)

  def test_to_uint8(self):
    self.assert_all_equal(
        media.to_uint8(np.array([0, 1, 128, 254, 255], dtype=np.uint8)),
        [0, 1, 128, 254, 255],
    )
    self.assert_all_close(
        media.to_uint8([-0.2, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1]),
        [
            0,
            0,
            int(0.1 * 255 + 0.5),
            int(0.5 * 255 + 0.5),
            int(0.9 * 255 + 0.5),
            255,
            255,
        ],
    )

  def test_color_ramp_float(self):
    shape = 2, 3
    image = media.color_ramp(shape=shape)
    self.assert_all_equal(image.shape[:2], shape)
    r1, r2 = (v / shape[0] for v in [0.5, 1.5])
    g1, g2, g3 = (v / shape[1] for v in [0.5, 1.5, 2.5])
    b = 0.0
    expected = [
        [[r1, g1, b], [r1, g2, b], [r1, g3, b]],
        [[r2, g1, b], [r2, g2, b], [r2, g3, b]],
    ]
    self.assert_all_close(image, expected)

  def test_color_ramp_uint8(self):
    shape = 1, 3
    image = media.color_ramp(shape=shape, dtype=np.uint8)
    self.assert_all_equal(image.shape[:2], shape)
    r = int(0.5 / shape[0] * 255 + 0.5)
    g1, g2, g3 = (int(v / shape[1] * 255 + 0.5) for v in [0.5, 1.5, 2.5])
    b = 0
    expected = [[[r, g1, b], [r, g2, b], [r, g3, b]]]
    self.assert_all_equal(image, expected)

  @parameterized.parameters(np.uint8, 'uint8', 'float32')
  def test_moving_circle(self, dtype):
    video = media.moving_circle(shape=(256, 256), num_images=10, dtype=dtype)
    self.assert_all_equal(video.shape, (10, 256, 256, 3))
    mean_image = np.mean(video, axis=0)
    expected_mean = 0.329926 if dtype == 'float32' else 84.295
    self.assertAlmostEqual(np.std(mean_image), expected_mean, delta=0.001)

  def test_rgb_yuv_roundtrip(self):
    image = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [255, 255, 255],
            [128, 128, 128],
        ],
        dtype=np.uint8,
    )
    new = media.to_uint8(media.rgb_from_yuv(media.yuv_from_rgb(image)))
    self.assert_all_close(image, new, atol=1)

  def test_rgb_ycbcr_roundtrip(self):
    image = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [255, 255, 255],
            [128, 128, 128],
        ],
        dtype=np.uint8,
    )
    new = media.to_uint8(media.rgb_from_ycbcr(media.ycbcr_from_rgb(image)))
    self.assert_all_close(image, new, atol=1)

  def test_pil_image(self):
    im = media._pil_image(
        np.array([[[10, 11, 12], [40, 41, 42]]], dtype=np.uint8)
    )
    self.assertEqual(im.width, 2)
    self.assertEqual(im.height, 1)
    self.assertEqual(im.mode, 'RGB')
    a = np.array(im)
    self.assert_all_equal(a.shape, (1, 2, 3))
    self.assert_all_equal(a, [[[10, 11, 12], [40, 41, 42]]])

  @parameterized.parameters(zip(_TEST_TYPES, _TEST_SHAPES1))
  def test_resize_image(self, str_dtype, shape):
    dtype = np.dtype(str_dtype)

    def create_image(shape):
      image = media.color_ramp(shape[:2], dtype=dtype)
      return (
          image.mean(axis=-1).astype(dtype)
          if len(shape) == 2
          else image[..., : shape[2]]
      )

    image = create_image(shape)
    self.assertEqual(image.dtype, dtype)
    new_shape = (17, 19) + shape[2:]
    new_image = media.resize_image(image, new_shape[:2])
    self.assertEqual(new_image.dtype, dtype)
    expected_image = create_image(new_shape)
    atol = 0.0 if new_shape == shape else 0.015
    self.assert_all_close(
        media.to_float01(new_image), media.to_float01(expected_image), atol=atol
    )

  @parameterized.parameters(zip(_TEST_TYPES, _TEST_SHAPES2))
  def test_resize_video(self, str_dtype, shape):
    dtype = np.dtype(str_dtype)

    def create_video(shape, num_images=5):
      video = media.moving_circle(shape[:2], num_images, dtype=dtype)
      return (
          video.mean(axis=-1).astype(dtype)
          if len(shape) == 2
          else video[..., : shape[2]]
      )

    video = create_video(shape)
    self.assertEqual(video.dtype, dtype)
    new_shape = (17, 19) + shape[2:]
    new_video = media.resize_video(video, new_shape[:2])
    self.assertEqual(new_video.dtype, dtype)
    expected_video = create_video(new_shape)
    self._check_similar(
        media.to_float01(new_video),
        media.to_float01(expected_video),
        max_rms=(0.0 if new_shape == shape else 0.07),
    )

  def test_read_contents(self):
    data = b'Test data'
    with tempfile.TemporaryDirectory() as directory_name:
      tmp_path = pathlib.Path(directory_name) / 'file'
      tmp_path.write_bytes(data)
      new_data = media.read_contents(tmp_path)
      self.assertEqual(new_data, data)
      new_data = media.read_contents(str(tmp_path))
      self.assertEqual(new_data, data)

  def test_read_via_local_file_on_local_file(self):
    with tempfile.TemporaryDirectory() as directory_name:
      tmp_path = pathlib.Path(directory_name) / 'file'
      tmp_path.write_text('text', encoding='utf-8')
      with media._read_via_local_file(tmp_path) as local_filename:
        self.assertEqual(local_filename, str(tmp_path))

  def test_write_via_local_file_on_local_file(self):
    with tempfile.TemporaryDirectory() as directory_name:
      tmp_path = pathlib.Path(directory_name) / 'file'
      with media._write_via_local_file(tmp_path) as local_filename:
        self.assertEqual(local_filename, str(tmp_path))

  @parameterized.parameters('uint8', 'uint16')
  def test_image_write_read_roundtrip(self, dtype):
    image = media.color_ramp((27, 63), dtype=dtype)
    self.assertEqual(image.dtype, dtype)
    if dtype == 'uint16':
      # Unfortunately PIL supports only single-channel 16-bit images for now.
      image = image[..., 0]
    with tempfile.TemporaryDirectory() as directory_name:
      path = pathlib.Path(directory_name) / 'test.png'
      media.write_image(path, image)
      new_image = media.read_image(path)
      self.assert_all_equal(image.shape, new_image.shape)
      self.assertEqual(image.dtype, new_image.dtype)
      self.assert_all_equal(image, new_image)

  def test_to_rgb(self):
    a = np.array([[-0.2, 0.0, 0.2, 0.8, 1.0, 1.2]])

    def gray_color(x):
      return [x, x, x]

    expected = [[gray_color(v / 1.4) for v in [0.0, 0.2, 0.4, 1.0, 1.2, 1.4]]]
    self.assert_all_close(media.to_rgb(a), expected, atol=0.002)

    expected = [[gray_color(v) for v in [0.0, 0.0, 0.2, 0.8, 1.0, 1.0]]]
    self.assert_all_close(
        media.to_rgb(a, vmin=0.0, vmax=1.0), expected, atol=0.002
    )

    a = np.array([-0.4, 0.0, 0.2])
    self.assert_all_close(
        media.to_rgb(a, vmin=-1.0, vmax=1.0, cmap='bwr'),
        [[0.596078, 0.596078, 1.0], [1.0, 0.996078, 0.996078], [1.0, 0.8, 0.8]],
        atol=0.002,
    )

  def test_uint8_to_rgb(self):
    a = np.array([100, 120, 140], dtype=np.uint8)

    def gray(x):
      if hasattr(matplotlib, 'colormaps'):
        cmap = matplotlib.colormaps['gray']  # Newer version.
      else:
        cmap = matplotlib.pyplot.cm.get_cmap('gray')
      return cmap(x)[..., :3]

    self.assert_all_close(media.to_rgb(a), gray([0.0, 0.5, 1.0]))
    self.assert_all_close(
        media.to_rgb(a, vmin=80, vmax=160), gray([0.25, 0.5, 0.75])
    )
    self.assert_all_close(
        media.to_rgb(a, vmin=110, vmax=160), gray([0.0, 0.2, 0.6])
    )
    self.assert_all_close(
        media.to_rgb(a, vmin=110, vmax=130), gray([0.0, 0.5, 1.0])
    )

  @parameterized.parameters('uint8', 'uint16')
  def test_compress_decompress_image_roundtrip(self, dtype):
    image = media.color_ramp((27, 63), dtype=dtype)
    if dtype == 'uint16':
      # Unfortunately PIL supports only single-channel 16-bit images for now.
      image = image[..., 0]
    data = media.compress_image(image)
    new_image = media.decompress_image(data, dtype=dtype)
    self.assertEqual(image.shape, new_image.shape)
    self.assertEqual(image.dtype, new_image.dtype)
    self.assert_all_equal(image, new_image)

  def test_show_image(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_image(media.color_ramp())
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<table', htmls[0].data), 1)
    self.assertRegex(htmls[0].data, '(?s)<img width=[^<>]*/>')
    self.assertLen(re.findall('(?s)<img', htmls[0].data), 1)

  def test_set_show_save_dir_image(self):
    self.enter_context(mock.patch('IPython.display.display'))
    with tempfile.TemporaryDirectory() as directory_name:
      directory_path = pathlib.Path(directory_name)
      with media.set_show_save_dir(directory_path):
        media.show_images({'ramp1': media.color_ramp((128, 128))})
      path = directory_path / 'ramp1.png'
      self.assertTrue(path.is_file())
      self.assertBetween(path.stat().st_size, 200, 1000)

      media.show_images({'ramp2': media.color_ramp((128, 128))})
      self.assertFalse((directory_path / 'ramp2.png').is_file())

      media.set_show_save_dir(directory_path)
      media.show_images({'ramp3': media.color_ramp((128, 128))})
      self.assertTrue((directory_path / 'ramp3.png').is_file())

      media.set_show_save_dir(None)
      media.show_images({'ramp4': media.color_ramp((128, 128))})
      self.assertFalse((directory_path / 'ramp4.png').is_file())

  def test_show_image_downsampled(self):
    image = np.random.default_rng(0).random((256, 256, 3))
    for downsample in (False, True):
      htmls = []
      with mock.patch('IPython.display.display', htmls.append):
        media.show_image(image, height=64, downsample=downsample)
      size_min_max = (10_000, 20_000) if downsample else (200_000, 300_000)
      self.assertBetween(len(htmls[0].data), *size_min_max)

  def test_show_image_default_no_pixelated(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_image(media.color_ramp((10, 10)))
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<img', htmls[0].data), 1)
    self.assertLen(re.findall('(?s)image-rendering:auto', htmls[0].data), 1)
    self.assertEmpty(re.findall('(?s)image-rendering:pixelated', htmls[0].data))

  def test_show_image_magnified_pixelated(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_image(media.color_ramp((10, 10)), width=20)
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<img', htmls[0].data), 1)
    self.assertLen(
        re.findall('(?s)image-rendering:pixelated', htmls[0].data), 1
    )
    self.assertEmpty(re.findall('(?s)image-rendering:auto', htmls[0].data))

  def test_show_images_list(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_images([media.color_ramp()] * 2)
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<table', htmls[0].data), 1)
    self.assertLen(re.findall('(?s)<img', htmls[0].data), 2)

  def test_show_images_dict(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_images(
          {'title1': media.color_ramp(), 'title2': media.color_ramp()}
      )
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<table', htmls[0].data), 1)
    self.assertRegex(htmls[0].data, '(?s)title1.*<img .*title2.*<img ')
    self.assertLen(re.findall('(?s)<img', htmls[0].data), 2)

  def test_show_images_over_multiple_rows(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_images([media.color_ramp()] * 5, columns=2)
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<table', htmls[0].data), 3)
    self.assertLen(re.findall('(?s)<img', htmls[0].data), 5)

  def test_compare_images(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.compare_images([media.color_ramp()] * 2)
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<img-comparison-slider>', htmls[0].data), 1)
    self.assertLen(re.findall('(?s)base64', htmls[0].data), 2)
    self.assertEmpty(re.findall('(?s)b64', htmls[0].data))

  @parameterized.parameters(False, True)
  def test_video_non_streaming_write_read_roundtrip(self, use_generator):
    shape = 240, 320
    num_images = 10
    fps = 40
    qp = 20
    original_video = media.to_uint8(media.moving_circle(shape, num_images))
    video = (
        (image for image in original_video) if use_generator else original_video
    )
    with tempfile.TemporaryDirectory() as directory_name:
      tmp_path = pathlib.Path(directory_name) / 'test.mp4'
      media.write_video(tmp_path, video, fps=fps, qp=qp)
      new_video = media.read_video(tmp_path)
      assert new_video.metadata
      self.assertEqual(new_video.metadata.num_images, num_images)
      self.assertEqual(new_video.metadata.shape, shape)
      self.assertEqual(new_video.metadata.fps, fps)
      self.assertGreater(new_video.metadata.bps, 1_000)
      self._check_similar(original_video, new_video, 3.0)

  def test_video_streaming_write_read_roundtrip(self):
    shape = 62, 744
    num_images = 20
    fps = 120
    bps = 400_000
    with tempfile.TemporaryDirectory() as directory_name:
      tmp_path = pathlib.Path(directory_name) / 'test.mp4'
      images = []
      with media.VideoWriter(tmp_path, shape, fps=fps, bps=bps) as writer:
        for image in media.moving_circle(shape, num_images):
          image_uint8 = media.to_uint8(image)
          writer.add_image(image_uint8)
          images.append(image_uint8)

      with media.VideoReader(tmp_path) as reader:
        self.assertEqual(reader.num_images, num_images)
        self.assert_all_equal(reader.shape, shape)
        self.assertEqual(reader.fps, fps)
        self.assertBetween(reader.bps, 100_000, 500_000)
        self.assertEqual(reader.metadata.num_images, reader.num_images)
        self.assertEqual(reader.metadata.shape, reader.shape)
        self.assertEqual(reader.metadata.fps, reader.fps)
        self.assertEqual(reader.metadata.bps, reader.bps)
        for index, new_image in enumerate(reader):
          self._check_similar(images[index], new_image, 7.0, f'index={index}')

  def test_video_streaming_read_write(self):
    shape = 400, 400
    num_images = 4
    fps = 25
    bps = 40_000_000
    video = media.to_uint8(media.moving_circle(shape, num_images))
    with tempfile.TemporaryDirectory() as directory_name:
      tmp_dir = pathlib.Path(directory_name)
      path1 = tmp_dir / 'test1.mp4'
      path2 = tmp_dir / 'test2.mp4'
      media.write_video(path1, video, fps=fps, bps=bps)

      with media.VideoReader(path1) as reader:
        with media.VideoWriter(
            path2,
            reader.shape,
            metadata=reader.metadata,
            encoded_format='yuv420p',
        ) as writer:
          for image in reader:
            writer.add_image(image)

      new_video = media.read_video(path2)
      assert new_video.metadata
      self.assertEqual(new_video.metadata.num_images, num_images)
      self.assertEqual(new_video.metadata.shape, shape)
      self.assertEqual(new_video.metadata.fps, fps)
      self.assertGreater(new_video.metadata.bps, 1_000)
      self._check_similar(video, new_video, 3.0)

  def test_video_read_write_10bit(self):
    shape = 256, 256
    num_images = 4
    fps = 60
    bps = 40_000_000
    horizontal_gray_ramp = media.to_type(
        np.indices(shape)[1] / shape[1], np.uint16
    )
    video = np.broadcast_to(horizontal_gray_ramp, (num_images, *shape))
    with tempfile.TemporaryDirectory() as directory_name:
      path = pathlib.Path(directory_name) / 'test3.mp4'
      media.write_video(
          path, video, fps=fps, bps=bps, encoded_format='yuv420p10le'
      )
      new_video = media.read_video(path, dtype=np.uint16, output_format='gray')
    self.assertEqual(new_video.dtype, np.uint16)
    value_1_of_10bit_encoded_in_16bits = 64
    self._check_similar(
        video, new_video, max_rms=value_1_of_10bit_encoded_in_16bits * 0.8
    )

  def test_video_read_write_vp9(self):
    video = media.moving_circle((256, 256), num_images=4, dtype=np.uint8)
    fps = 60
    bps = 40_000_000
    with tempfile.TemporaryDirectory() as directory_name:
      path = pathlib.Path(directory_name) / 'test4.mp4'
      media.write_video(path, video, fps=fps, bps=bps, codec='vp9')
      new_video = media.read_video(path)
    self.assertEqual(new_video.dtype, np.uint8)
    self._check_similar(video, new_video, max_rms=5.0)

  def test_video_read_write_odd_dimensions(self):
    video = media.moving_circle((35, 97), num_images=4, dtype=np.uint8)
    fps = 60
    bps = 40_000_000
    with tempfile.TemporaryDirectory() as directory_name:
      path = pathlib.Path(directory_name) / 'test5.mp4'
      media.write_video(path, video, fps=fps, bps=bps)
      new_video = media.read_video(path)
    self.assertEqual(new_video.dtype, np.uint8)
    self._check_similar(video, new_video, max_rms=5.0)

  def test_compress_decompress_video_roundtrip(self):
    video = media.moving_circle((28, 66), num_images=10, dtype=np.uint8)
    data = media.compress_video(video)
    new_video = media.decompress_video(data)
    self.assertEqual(video.shape, new_video.shape)
    self.assertEqual(video.dtype, new_video.dtype)
    self._check_similar(video, new_video, max_rms=8.0)

  def test_html_from_compressed_video(self):
    shape = 240, 320
    video = media.moving_circle(shape, 10)
    text = media.html_from_compressed_video(
        media.compress_video(video), shape[1], shape[0]
    )
    self.assertGreater(len(text), 2_000)
    self.assertContainsInOrder(
        ['<video', '<source src="data', 'type="video/mp4"/>', '</video>'], text
    )

  def test_show_video(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_video(media.moving_circle())
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<video', htmls[0].data), 1)
    self.assertRegex(htmls[0].data, '(?s)<video .*>.*</video>')

  def test_show_video_gif(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_video(media.moving_circle(), codec='gif')
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertContainsInOrder(['<img', 'src="data:image/gif'], htmls[0].data)

  def test_set_show_save_dir_video(self):
    video = media.moving_circle((32, 32), num_images=10)
    with tempfile.TemporaryDirectory() as directory_name:
      directory_path = pathlib.Path(directory_name)
      with media.set_show_save_dir(directory_path):
        with mock.patch('IPython.display.display'):
          media.show_videos({'video0': video, 'video1': video})
      for i in range(2):
        path = directory_path / f'video{i}.mp4'
        self.assertTrue(path.is_file())
        self.assertBetween(path.stat().st_size, 1500, 3000)

  def test_show_video_downsampled(self):
    video = np.random.default_rng(0).random((5, 64, 128, 3))
    for downsample in (False, True):
      htmls = []
      with mock.patch('IPython.display.display', htmls.append):
        media.show_video(video, height=32, downsample=downsample)
      size_min_max = (8_000, 15_000) if downsample else (40_000, 60_000)
      self.assertBetween(len(htmls[0].data), *size_min_max)

  def test_show_videos_list(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_videos([media.moving_circle()] * 2)
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<table', htmls[0].data), 1)
    self.assertLen(re.findall('(?s)<video', htmls[0].data), 2)

  def test_show_videos_dict(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      videos = {
          'title1': media.moving_circle(),
          'title2': media.moving_circle(),
      }
      media.show_videos(videos)
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<table', htmls[0].data), 1)
    self.assertRegex(htmls[0].data, '(?s)title1.*<video.*title2.*<video')
    self.assertLen(re.findall('(?s)<video', htmls[0].data), 2)

  def test_show_videos_over_multiple_rows(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      media.show_videos([media.moving_circle()] * 12, columns=3)
    self.assertLen(htmls, 1)
    self.assertIsInstance(htmls[0], IPython.display.HTML)
    self.assertLen(re.findall('(?s)<table', htmls[0].data), 4)
    self.assertLen(re.findall('(?s)<video', htmls[0].data), 12)

  def test_show_image_out_str(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      out = media.show_image(media.color_ramp(), return_html=True)
    self.assertEmpty(htmls)  # Nothing displayed
    self.assertIsInstance(out, str)
    self.assertIn('<img', out)

  def test_show_images_out_str(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      out = media.show_images([media.color_ramp()] * 3, return_html=True)
    self.assertEmpty(htmls)  # Nothing displayed
    self.assertIsInstance(out, str)
    self.assertIn('<img', out)

  def test_show_video_out_str(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      out = media.show_video(media.moving_circle(), return_html=True)
    self.assertEmpty(htmls)  # Nothing displayed
    self.assertIsInstance(out, str)
    self.assertIn('<video', out)

  def test_show_videos_out_str(self):
    htmls = []
    with mock.patch('IPython.display.display', htmls.append):
      out = media.show_videos([media.moving_circle()] * 2, return_html=True)
    self.assertEmpty(htmls)  # Nothing displayed
    self.assertIsInstance(out, str)
    self.assertIn('<video', out)


if __name__ == '__main__':
  absltest.main()


# Local Variables:
# fill-column: 80
# End:
