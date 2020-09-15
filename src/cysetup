from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("update_weight", sources=["update_weight.pyx"], include_dirs=[get_include()])
setup(name="update_weight", ext_modules=cythonize([ext]), include_dirs=[get_include()])
