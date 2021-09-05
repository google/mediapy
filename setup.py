# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Install mediapy package."""

import pathlib
import setuptools


def get_long_description():
  return pathlib.Path('README.md').read_text()


def get_version(rel_path):
  path = pathlib.Path(__file__).resolve().parent / rel_path
  for line in path.read_text().splitlines():
    if line.startswith("__version__ = '"):
      _, version, _ = line.split("'")
      return version
  raise RuntimeError(f'Unable to find version string in {path}.')


setuptools.setup(
    name='mediapy',
    version=get_version('mediapy/__init__.py'),
    author='Google LLC',
    author_email='mediapy-owners@google.com',
    description='Read/write/show images and videos in an IPython notebook',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/google/mediapy',
    license='Apache-2.0',
    packages=['mediapy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'Topic :: Multimedia :: Video :: Display',
    ],
    python_requires='>=3.6',
    install_requires=[
        'ipython',
        'matplotlib',
        'numpy',
        'Pillow',
    ],
)
