from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
# include_dirs_numpy = [numpy.get_include()]

setup(
    ext_modules=cythonize("fplib3.pyx"),
    include_dirs=[numpy.get_include()]
)  
