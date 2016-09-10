from distutils.core import setup
from Cython.Build import cythonize

setup(
        ext_modules=cythonize([
            "mini_hmm_cython.pyx",
            "mini_ziphmm_cython_funcs.pyx",
            ])
)
