#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pathlib import Path

from setuptools import find_packages, setup


# Package meta-data.
NAME = 'libreface'
DESCRIPTION = 'LibreFace model for facial analysis'
URL = 'https://boese0601.github.io/libreface'
EMAIL = 'achaubey@usc.edu'
AUTHOR = 'IHP-Lab'
REQUIRES_PYTHON = '>=3.8.0'


# What packages are required for this module to be executed?
def list_reqs(fname='requirements_new.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the
# Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with open("README_pypi.rst") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

try:
    with open("LICENSE.rst") as f:
        license = "USC Research Licence \n"+f.read()
except FileNotFoundError:
    license = "USC"

# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
# with open(PACKAGE_DIR / 'VERSION') as f:
#     _version = f.read().strip()

about['__version__'] = "0.0.14"


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description='LibreFace model for facial analysis',
    long_descripation_content_type='text/x-rst',
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    package_data={'libreface': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license=license,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)