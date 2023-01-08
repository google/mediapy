#!/usr/bin/env python3
"""Create HTML documentation from the source code using `pdoc`."""

# Note: Invoke this script from the parent directory as "pdoc/make_pdoc.py".

import pathlib
import re

import pdoc

if 0:  # https://github.com/mitmproxy/pdoc/issues/420
  pdoc.doc_types.simplify_annotation.replacements['AAAAAA'] = 'B'
  pdoc.doc_types.simplify_annotation.recompile()

MODULES = ['./mediapy']  # One or more module names or file paths.
FAVICON = 'https://github.com/google/mediapy/raw/main/pdoc/favicon.ico'
FOOTER_TEXT = ''
LOGO = 'https://github.com/google/mediapy/raw/main/pdoc/logo.png'
LOGO_LINK = 'https://google.github.io/mediapy/'
TEMPLATE_DIRECTORY = pathlib.Path('./pdoc')
OUTPUT_DIRECTORY = pathlib.Path('./docs')


def main() -> None:
  """Invoke `pdoc` on the module source files."""
  # See https://github.com/mitmproxy/pdoc/blob/main/pdoc/__main__.py
  pdoc.render.configure(
    docformat='restructuredtext',  # 'google' is inferred from __docformat__.
    edit_url_map=None,
    favicon=FAVICON,
    footer_text=FOOTER_TEXT,
    logo=LOGO,
    logo_link=LOGO_LINK,
    math=True,  # Default is False.
    search=True,  # Default is True.
    show_source=True,  # Default is True.
    template_directory=TEMPLATE_DIRECTORY,
  )

  pdoc.pdoc(
    *MODULES,
    output_directory=OUTPUT_DIRECTORY,
  )

  if 1:
    output_file = OUTPUT_DIRECTORY / 'mediapy.html'
    text = output_file.read_text()
    # e.g., collections.abc.Iterable -> Iterable.
    text = text.replace('<span class="n">collections</span><span class="o">'
                        '.</span><span class="n">abc</span><span class="o">.</span>', '')
    # typing.* -> *.
    text = text.replace('<span class="n">typing</span><span class="o">.</span>', '')
    output_file.write_text(text)


if __name__ == '__main__':
  main()
