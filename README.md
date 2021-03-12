# `mediapy`: Read/write/show images and videos in an IPython notebook.

## Examples:

See extensive examples in the notebook
[`mediapy_examples.ipynb`](https://github.com/google/mediapy/blob/main/mediapy_examples.ipynb)
&mdash;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/google/mediapy/blob/main/mediapy_examples.ipynb).

### Images:

```python
    import mediapy as media
    import numpy as np

    image = media.read_image('https://github.com/hhoppe/data/raw/main/image.png')
    print(image.shape, image.dtype)  # It is a numpy array.
    media.show_image(image)

    checkerboard = np.kron([[0, 1] * 16, [1, 0] * 16] * 16, np.ones((4, 4)))
    media.show_image(checkerboard)

    media.show_image(media.color_ramp((128, 128)), height=48, title='ramp')

    images = {
        'original': media.color_ramp(),
        'brightened': media.color_ramp() * 1.5,
    }
    media.show_images(images)

    media.write_image('/tmp/checkerboard.png', checkerboard)
```

### Videos:

```python
    video = media.read_video('https://github.com/hhoppe/data/raw/main/video.mp4')
    print(video.shape, video.dtype)  # It is a numpy array.
    media.show_video(video)

    media.show_images(video, columns=4)  # Show the video frames side-by-side.

    video = media.moving_circle((128, 128), num_images=10)
    media.show_video(video)

    media.write_video('/tmp/video.mp4', video)

    # Darken a video frame-by-frame:
    filename_in = '/tmp/video.mp4'
    filename_out = '/tmp/out.mp4'
    with media.VideoReader(filename_in) as r:
      print(f'shape={r.shape} fps={r.fps} bps={r.bps}')
      darken_image = lambda image: media.to_float01(image) * 0.5
      with media.VideoWriter(
          filename_out, shape=r.shape, fps=r.fps, bps=r.bps) as w:
        for image in r:
          w.add_image(darken_image(image))
    media.show_video(media.read_video(filename_out))
```

## Setup:

Video I/O relies on the external program `ffmpeg`, which must be present in
the system PATH.  On Unix, it can be installed using:

```shell
    apt-get install ffmpeg
```
