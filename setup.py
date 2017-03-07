from setuptools import find_packages, setup
from Cython.Build import cythonize
import os
from codecs import open
import numpy

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ziphmm",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize([
        "ziphmm/cython_funcs.pyx",
        ]),
    include_dirs=[numpy.get_include()],

    # metadata for upload to PyPI
    author="Anders Egerup Halager",
    author_email="aeh@birc.au.dk",
    license="MIT",
    keywords="hmm zip ziphmm",
    url="https://github.com/birc-aeh/mini-ziphmm",
    description="An implementation of the ziphmm algorithm",
    long_description=long_description,

    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
)
