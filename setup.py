from glob import glob
from os.path import basename
from os.path import splitext

from distutils.core import setup

from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension
from numpy import get_include # cimport numpy を使うため


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name='hpneat',
    version='0.0.1',
    license='ライセンス',
    description='パッケージの説明',
    author='作成者',
    url='GitHubなどURL',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    include_dirs=[get_include()]
    #install_requires=_requires_from_file('requirements.txt'),
    zip_safe=False,
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
    ext_modules=[
        Extension('update_weight',sources=['src/update_weight.pyx'])
    ]
)
