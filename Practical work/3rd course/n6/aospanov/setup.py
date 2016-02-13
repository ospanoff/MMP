from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy

sourcefiles = ['cy_cluster.pyx', 'k_medoids.cpp']

extensions = [Extension(
    'cy_cluster', sourcefiles, language='c++', include_dirs=[numpy.get_include()], extra_compile_args=["-std=c++11"])]

setup(
    ext_modules=cythonize(extensions)
)
