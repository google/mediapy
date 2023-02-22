#!/usr/bin/env python3
"""Create HTML documentation from the source code using `pdoc`."""
# Note: Invoke this script from the parent directory as "pdoc_files/make.py".

import pathlib
import re

import pdoc

MODULE_NAME = 'mediapy'
FAVICON = 'https://github.com/google/mediapy/raw/main/pdoc_files/favicon.ico'
FOOTER_TEXT = ''
LOGO = 'https://github.com/google/mediapy/raw/main/pdoc_files/logo.png'
LOGO_LINK = 'https://google.github.io/mediapy/'
TEMPLATE_DIRECTORY = pathlib.Path('./pdoc_files')
OUTPUT_DIRECTORY = pathlib.Path('./pdoc_files/html')
APPLY_POSTPROCESS = True


def main() -> None:
  """Invoke `pdoc` on the module source files."""
  # See https://github.com/mitmproxy/pdoc/blob/main/pdoc/__main__.py
  pdoc.render.configure(
      docformat='google',
      edit_url_map=None,
      favicon=FAVICON,
      footer_text=FOOTER_TEXT,
      logo=LOGO,
      logo_link=LOGO_LINK,
      math=True,
      search=True,
      show_source=True,
      template_directory=TEMPLATE_DIRECTORY,
  )

  pdoc.pdoc(
      f'./{MODULE_NAME}',
      output_directory=OUTPUT_DIRECTORY,
  )

  if APPLY_POSTPROCESS:
    output_file = OUTPUT_DIRECTORY / f'{MODULE_NAME}.html'
    text = output_file.read_text()

    # collections.abc.* -> * (Iterable, Mapping, Callable, etc.).
    text = text.replace(
        (
            '<span class="n">collections</span><span class="o">'
            '.</span><span class="n">abc</span><span class="o">.</span>'
        ),
        '',
    )

    # typing.* -> * (e.g. typing.Any).
    text = text.replace(
        '<span class="n">typing</span><span class="o">.</span>',
        '',
    )

    # ~_ArrayLike -> ArrayLike.
    text = re.sub(
        r'(?s)<span class="o">~</span>\s*<span class="n">_ArrayLike',
        r'<span class="n">ArrayLike',
        text,
    )
    text = text.replace('~_ArrayLike', 'ArrayLike')

    # ~_DTypeLike -> DTypeLike.
    text = re.sub(
        r'(?s)<span class="o">~</span>\s*<span class="n">_DTypeLike',
        r'<span class="n">DTypeLike',
        text,
    )
    text = text.replace('~_DTypeLike', 'DTypeLike')

    # ~_NDArray -> np.ndarray.
    text = re.sub(
        r'(?s)<span class="o">~</span>\s*<span class="n">_NDArray<',
        r'<span class="n">np.ndarray<',
        text,
    )
    text = text.replace('~_NDArray', 'np.ndarray')

    # ~_DType -> np.dtype.
    text = re.sub(
        r'(?s)<span class="o">~</span>\s*<span class="n">_NDType<',
        r'<span class="n">np.dtype<',
        text,
    )
    text = text.replace('~_NDType', 'np.dtype')

    output_file.write_text(text)


if __name__ == '__main__':
  main()

# Local Variables:
# fill-column: 80
# End:
