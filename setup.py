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
    packages=['AutoEncoder',
              'GAM'],
    version='0.0.1',
    description="Data Science Toolkit",
    url='https://github.com/jotterbach/dstk',
    install_requires=[
        'numpy',
        'tensorflow',
        'sklearn',
        'pandas'
    ],
    test_requires=[
        'pytest'
    ],
    dependency_links=[
        "https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl"
    ]
)
