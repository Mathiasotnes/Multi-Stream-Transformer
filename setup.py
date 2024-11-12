##############################################################################
## setup.py                                                                 ##
## --------                                                                 ##
## Setup file for the Multi-Stream Transformer package.                     ##
##                                                                          ##
## ------------------------------------------------------------------------ ##
## Author:   Mathias Otnes                                                  ##
## year:     2024                                                           ##
##############################################################################

from setuptools import setup, find_packages

setup(
    name='MSTransformer',
    version='0.1',
    packages=find_packages(),
    install_requires=[ # Should read from requirements.txt
        'torch',
        'numpy',
        'matplotlib'
    ]
)
