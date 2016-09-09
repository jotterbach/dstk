#! /usr/bin/env python
# -*- coding: utf-8 -*-

import DSTK

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    author="Johannes Otterbach",
    author_email="johannesotterbach@gmail.com",
    name="DSTK",
    packages=['DSTK'],
    version=DSTK.__version__,
    description="Data Science Toolkit",
    url='https://github.com/jotterbach/dstk',
    install_requires=[
        'numpy',
        'tensorflow',
        'sklearn',
        'pandas',
        'fuzzywuzzy',
        'futures',
        'statsmodels'
    ],
    tests_require=[
        'pytest'
    ],
    test_suite='DSTK.tests',
    dependency_links=[
        "https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl"
    ]
)
