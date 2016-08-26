#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    author="Johannes Otterbach",
    author_email="johannesotterbach@gmail.com",
    name="dstk",
    packages=['DSTK'],
    version='0.0.1',
    description="Data Science Toolkit",
    url='https://github.com/jotterbach/dstk',
    install_requires=[
        'numpy',
        'tensorflow',
        'sklearn',
        'pandas',
        'fuzzywuzzy'
    ],
    tests_require=[
        'pytest'
    ],
    test_suite='DSTK.tests',
    dependency_links=[
        "https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl"
    ]
)
