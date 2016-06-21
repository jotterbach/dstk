#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

version = "1.0"

setup(
    author="Johannes Otterbach",
    author_email="johannesotterbach@gmail.com",
    name="AutoEncoder",
    packages=['AutoEncoder',
              'AutoEncoder'],
    version='0.0.1',
    description="Simple Autoencoder built with TensorFlow",
    url='https://github.com/jotterbach/AutoEncoder',
    install_requires=[
        'numpy',
        'tensorflow'
    ],
    dependency_links=[
        "https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl"
    ]
)
