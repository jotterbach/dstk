#! /usr/bin/env python
# -*- coding: utf-8 -*-

import DSTK
from setuptools import setup, find_packages


setup(
    author="Johannes Otterbach",
    author_email="johannesotterbach@gmail.com",
    name="DSTK",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version=DSTK.__version__,
    description="Data Science Toolkit",
    url='https://github.com/jotterbach/dstk',
    install_requires=[
        'numpy',
        'scipy',
        'tensorflow',
        'sklearn',
        'pandas',
        'fuzzywuzzy',
        'futures',
        'statsmodels',
        'patsy',
    ],
    tests_require=[
        'pytest'
    ],
    test_suite='DSTK.tests',
    dependency_links=[
        "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0rc0-py2-none-any.whl"
    ]
)
