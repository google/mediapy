# %% [markdown]
# # mediapy: Read/write/show images and videos in a Python notebook
#
# [**[GitHub source]**](https://github.com/google/mediapy) &nbsp;
# [**[API docs]**](https://google.github.io/mediapy/) &nbsp;
# [**[PyPI package]**](https://pypi.org/project/mediapy/) &nbsp;
# [**[Colab example]**](https://colab.research.google.com/github/google/mediapy/blob/main/mediapy_examples.ipynb)

# %% [markdown]
# ## Setup

# %%
# !command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
# !pip install -q mediapy

# %%
import itertools

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np

# pylint: disable=missing-function-docstring, redefined-outer-name

# %%
DATA_DIR = 'https://github.com/hhoppe/data/raw/main/'  # Or any local path.
IMAGE = DATA_DIR + 'image.png'
VIDEO = DATA_DIR + 'video.mp4'

# %% [markdown]
# ## Image examples

# %%
# Display an image (a 2D or 3D numpy array):
image = np.random.default_rng(1).random((5, 5, 3))
image1 = media.resize_image(image, (50, 50))
media.show_image(image1, border=True)

# %%
# Read, resize, and display an image:
image2a = media.read_image(IMAGE)
image2 = media.resize_image(image2a, (64, 64))
media.show_image(image2, title='A title')

# %%
# Show titled images side-by-side:
images = {
    'checkerboard': np.kron([[0, 1] * 8, [1, 0] * 8] * 8, np.ones((4, 4))),
    'darker noise': image1 * 0.7,
    'as YCbCr': media.ycbcr_from_rgb(image2),
    'as YUV': media.yuv_from_rgb(image2),
    'rotated': np.rot90(image2),
    'thresholded': media.yuv_from_rgb(image2)[..., 0] > 0.5,
}
media.show_images(images, vmin=0.0, vmax=1.0, border=True, height=100)

# %%
# Show a scalar image using a color map:
image = np.random.default_rng(1).random((5, 5)) - 0.5
image3 = media.resize_image(image, (60, 60))
media.show_image(image3, cmap='bwr', border=True)

# %%
# More examples of color maps and value bounds:
images = {
    'gray': image3,
    '[0, 1]': media.to_rgb(image3, vmin=0.0, vmax=1.0),
    'bwr': media.to_rgb(image3, cmap='bwr'),
    'jet': media.to_rgb(image3, cmap='jet'),
    'jet [0, 0.5]': media.to_rgb(image3, vmin=0.0, vmax=0.5, cmap='jet'),
    'radial': np.cos(((np.indices((60, 60)).T / 10) ** 2).sum(axis=-1)),
}
media.show_images(images, border=True, height=100)

# %%
# Compare two images using an interactive slider:
media.compare_images([image2a, media.yuv_from_rgb(image2a)[..., 0] > 0.5])

# %%
# Write an image to a file:
media.write_image('/tmp/image3.png', image3)

# %% [markdown]
# ## Video examples

# %%
# Display a video (a 3D or 4D array, or an iterable of images):
video1 = media.moving_circle((65, 65), num_images=10)
media.show_video(video1, fps=2)

# %%
# Show a video as a GIF, so that it is visible in the GitHub notebook preview:
media.show_video(1.0 - video1, height=48, codec='gif', fps=4)

# %%
# Show video frames side-by-side:
media.show_images(video1, columns=6, border=True, height=50)

# %%
# Show the frames with their indices:
media.show_images({f'{i}': image for i, image in enumerate(video1)}, width=50)

# %%
# Read, resize, and display a video:
video2 = media.read_video(VIDEO)
print(f'Shape is (num_images, height, width, num_channels) = {video2.shape}.')
if metadata := video2.metadata:
  print(f'Framerate is {metadata.fps} frames/s.')
video3 = media.resize_video(video2, tuple(np.array(video2.shape[1:3]) // 2))
media.show_video(video3, fps=5, codec='gif')

# %%
# Display a two-frame flipping video as a GIF:
media.show_video([image3 + 0.5, (image3 + 0.5) * 0.8], fps=2, codec='gif')


# %%
def darken_image(image):
  return media.to_float01(image) * 0.5


# %%
# Darken a video frame-by-frame:
new_file = '/tmp/out.mp4'
with media.VideoReader(VIDEO) as reader:
  print(f'num_images={reader.num_images} shape={reader.shape} fps={reader.fps}')
  with media.VideoWriter(
      new_file, shape=reader.shape, fps=reader.fps / 5
  ) as writer:
    for image in reader:
      writer.add_image(darken_image(image))

media.show_video(media.read_video(new_file), height=90)

# %%
# Show multiple videos side-by-side.
s = 90
videos = {
    'mirror loop': np.concatenate([video3, video3[::-1]], axis=0),
    'roll': (np.roll(media.color_ramp((s, s)), i, axis=0) for i in range(s)),
    'fade': (np.full((s, s), f) for f in np.linspace(0.0, 1.0, 50)),
}
media.show_videos(videos, fps=5)

# %%
# Write a video to a file:
media.write_video('/tmp/video1.mp4', video1, fps=10, qp=10)


# %% [markdown]
# ## Conway's Game of Life
#
# Cellular automaton implemented on a periodic domain.
# See [Wikipedia](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).
#
# Show the first 16 generations starting from a random configuration:


# %%
def game_of_life(shape=(40, 60), seed=1):
  grid = (np.random.default_rng(seed).random(shape) < 0.2).astype(np.int32)
  neighbors = set(itertools.product((-1, 0, 1), repeat=2)) - {(0, 0)}
  while True:
    yield grid == 0
    num_neighbors = np.add.reduce(
        [np.roll(grid, yx, axis=(0, 1)) for yx in neighbors]
    )
    grid = (num_neighbors == 3) | (grid & (num_neighbors == 2))


# %%
video = list(itertools.islice(game_of_life(), 16))
video = [video[0]] * 8 + video + [video[-1]] * 8  # Pause first and last frames.
media.show_video(video, height=160, fps=8, codec='gif', border=True)

# %% [markdown]
# Show the 100<sup>th</sup> generation starting from different random seeds:

# %%
images = {
    f'seed={seed}': next(itertools.islice(game_of_life(seed=seed), 100, None))
    for seed in range(10)
}
media.show_images(images, border=True, columns=5, height=80)

# %% [markdown]
# ## Mandelbrot Set
#
# Visualize divergence of $\|z_i\|$ in the sequence
# $z_0 = 0,~z_{i+1} = z_{i}^2 + c$ over complex numbers $c$.
# See [Wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set).


# %%
def mandelbrot(shape, center_xy=(-0.75, -0.5), radius=1.25, max_iter=200):
  yx = np.moveaxis(np.indices(shape), 0, -1)
  yx = (yx + 0.5 - np.array(shape) / 2) / max(shape) * 2  # in [-1, 1]^2
  c = np.dot(yx * radius + center_xy[::-1], (1j, 1))
  count_iter = np.zeros(shape)
  z = np.zeros_like(c)
  for it in range(max_iter):
    active = abs(z) < 4
    count_iter[active] = it
    z[active] = z[active] ** 2 + c[active]
  return np.where(active, 0, count_iter)


# %%
media.show_image(mandelbrot((200, 300)) ** 0.23, cmap='gnuplot2')


# %% [markdown]
# ## Format conversions


# %%
def apply_conversions(image):
  assert image.dtype == np.uint8
  for dtype in ('uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64'):
    a = media.to_type(image, dtype)
    print(f'dtype={a.dtype!s:<8} mean={a.mean()}')
    assert np.all(media.to_uint8(a) == image)


# %%
apply_conversions(image2)

# %% [markdown]
# ## Image rate-distortion


# %%
def analyze_image_rate_distortion(image, fmt='jpeg'):
  params = [10, 20, 40, 75, 90, 95, 100]
  images = {}
  num_bytes, psnrs = [], []

  for param in params:
    data = media.compress_image(image, fmt=fmt, quality=param)
    num_bytes.append(len(data))
    image_new = media.decompress_image(data)
    images[f'quality={param}'] = image_new[30:60, 0:30]
    rms_error = np.sqrt(np.mean(np.square(image_new - image)))
    psnr = 20 * np.log10(255.0 / rms_error)
    psnrs.append(psnr)

  media.show_images(images, border=True, ylabel=fmt, height=120)

  _, ax = plt.subplots(figsize=(10, 3.5))
  ax.plot(num_bytes, psnrs, 'o-', label=f'{fmt} (by quality parameter)')
  for x, y, param in zip(num_bytes, psnrs, params):
    kwargs = dict(textcoords='offset points', xytext=(0, 10), ha='center')
    ax.annotate(f'{param}', (x, y), **kwargs)

  ax.set_title('Image rate-distortion')
  ax.set_xlabel('Obtained size (bytes)')
  ax.set_ylabel('PSNR (dB)')
  ax.legend()
  ax.grid(True)
  plt.show()


# %%
analyze_image_rate_distortion(media.read_image(IMAGE))  # or try fmt='webp'


# %% [markdown]
# ## Video rate-distortion
#
# Note that the metadata in the video container may be a significant overhead
# for small videos; see https://superuser.com/questions/1617422/.


# %%
def analyze_video_rate_distortion(video, fps=30, codec='h264'):
  _, ax = plt.subplots(figsize=(10, 3.5))

  for encoded_format in ['yuv444p', 'yuv420p']:
    bitrates, psnrs = [], []
    for requested_mbps in [0.03, 0.1, 0.3, 1, 3, 10]:
      bps = int(requested_mbps * 1.0e6)
      data = media.compress_video(
          video, encoded_format=encoded_format, codec=codec, bps=bps, fps=fps
      )
      video_new = media.decompress_video(data)
      rms_error = np.sqrt(np.mean(np.square(video_new - video)))
      psnr = 20 * np.log10(255.0 / rms_error)
      obtained_bps = len(data) * 8 / (len(video) / fps)
      bitrates.append(obtained_bps)
      psnrs.append(psnr)
    ax.semilogx(bitrates, psnrs, 'o-', label=f'{codec}_{encoded_format}')

  ax.set_title('Video rate-distortion')
  ax.set_xlabel('Obtained bitrate (bits/s)')
  ax.set_ylabel('PSNR (dB)')
  ax.legend()
  ax.grid(True)
  plt.show()


# %%
analyze_video_rate_distortion(video3)

# %% [markdown]
# ## End
# <!--
# Local Variables:
# fill-column: 80
# End:
# -->
