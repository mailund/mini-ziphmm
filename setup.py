from distutils.core import setup
from Cython.Build import cythonize
import os
from codecs import open

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mini-ziphmm",
    version="0.0.0",
    setup_requires=["cython>=0.x"],
    ext_modules=cythonize([
        "mini_ziphmm_cython_funcs.pyx",
        ]),
    install_requires=["numpy","scipy"],

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
