#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : setup.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 5:18 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 5:18 下午 by shendu.ht  init
"""
import os

from setuptools import setup, find_packages


def read(file_name):
    absolute_path = os.path.abspath(os.path.dirname(__file__))
    return open(os.path.join(absolute_path, file_name))


setup(
    name='gaussian_process',
    version='1.1.0',
    url='https://github.com/shendu-ht/gaussian_processes',
    packages=find_packages(),
    author='shendu.ht',
    author_email='shendu.ht@outlook.com',
    description='bayesian optimization based on gaussian process',
    long_description=read('README.md'),
    install_requires=['numpy>=1.14.2', 'scipy>=1.0.1', 'matplotlib>=2.2.0'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    license='GPLv3'
)
