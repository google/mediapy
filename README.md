# Read/write/show images and videos in an IPython/Jupyter notebook.

[**[GitHub source]**](https://github.com/google/mediapy) &nbsp;
[**[API docs]**](https://google.github.io/mediapy/) &nbsp;
[**[PyPI package]**](https://pypi.org/project/mediapy/) &nbsp;
[**[Colab example]**](https://colab.research.google.com/github/google/mediapy/blob/main/mediapy_examples.ipynb)

## Examples:

See the notebook &nbsp;
[`mediapy_examples.ipynb`](https://github.com/google/mediapy/blob/main/mediapy_examples.ipynb)
&nbsp; &nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/mediapy/blob/main/mediapy_examples.ipynb)
&nbsp;
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/google/mediapy/main?filepath=mediapy_examples.ipynb)

<!--
DeepNote: The notebook runs correctly on https://deepnote.com/, but it cannot be
launched from GitHub with a single click.  Instead, one must:
- Start a notebook.
- Create a Terminal (console).
- Within the terminal, enter "git clone https://github.com/google/mediapy.git".
- Navigate to Files -> mediapy.
- Open the *.ipynb notebook.

Kaggle: The notebook also runs correctly on https://www.kaggle.com/ although
"pip install -q mediapy" requires first changing Settings -> Internet -> Enable,
which in turn requires a phone number verification.  Also, the notebook cannot
be launched from GitHub with a single click but must be manually uploaded as a
file.
-->

### Images:

```python
    !pip install -q mediapy
    import mediapy as media
    import numpy as np

    image = media.read_image('https://github.com/hhoppe/data/raw/main/image.png')
    print(image.shape, image.dtype)  # It is a numpy array.
    media.show_image(image)

    checkerboard = np.kron([[0, 1] * 16, [1, 0] * 16] * 16, np.ones((4, 4)))
    media.show_image(checkerboard)

    media.show_image(media.color_ramp((128, 128)), height=48, title='ramp')

    images = {
        'original': image,
        'brightened': media.to_float01(image) * 1.5,
    }
    media.show_images(images)

    media.write_image('/tmp/checkerboard.png', checkerboard)
```

### Videos:

```python
    url = 'https://github.com/hhoppe/data/raw/main/video.mp4'
    video = media.read_video(url)
    print(video.shape, video.dtype)  # It is a numpy array.
    print(video.metadata.fps)  # The 'metadata' attribute includes framerate.
    media.show_video(video)  # Play the video using the retrieved framerate.

    media.show_images(video, height=80, columns=4)  # Show frames side-by-side.

    video = media.moving_circle((128, 128), num_images=10)
    media.show_video(video, fps=10)

    media.write_video('/tmp/video.mp4', video, fps=60)

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
    media.show_video(media.read_video(filename_out), fps=60)
```

## Setup:

Video I/O relies on the external program `ffmpeg`, which must be present in
the system PATH.  On Unix, it can be installed using:

```shell
    apt-get install ffmpeg
```

or within a notebook using:

```python
    !command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
```
