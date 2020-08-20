import os
import shutil
import sys

from setuptools import setup, find_packages, Command
from os import path

NAME = 'mpwmi'
DESCRIPTION = 'Message Passing Weighted Model Integration.'
URL = 'https://github.com/UCLA-StarAI/mpwmi'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = "0.1"

# What packages are required for this module to be executed?
REQUIRED = [
    'pysmt', 'numpy', 'sympy', 'networkx',
]



setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    python_requires=REQUIRES_PYTHON,
    zip_safe=False,
    install_requires=REQUIRED
)
