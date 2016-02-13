from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy

sourcefiles = ['fast.pyx', 'fast_impl.cpp']

extensions = [Extension(
    'fast', sourcefiles, language='c++', include_dirs=[numpy.get_include()])]

setup(
    ext_modules=cythonize(extensions)
)
