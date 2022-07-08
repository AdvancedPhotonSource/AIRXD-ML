import setuptools
from os import path
from distutils.core import setup

droot = path.abspath(path.dirname(__file__))
with open(path.join(droot, 'README.md'), encoding="utf-8") as f:
    long_description = f.read()
with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open("airxd/version.py").read())

setup(
    name='airxd',
    version=__version__,
    description='ML application for 2D XRD data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["airxd"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    #package_data={
    #    'airxd': ["mask.cpp"],
    #},
    package_dir={'airxd': 'airxd'},
    install_requires=[
        'cffi>=1.15.1',
        'numpy==1.21.6',
        'scipy==1.7.3',
        'matplotlib==3.5.2',
        'scikit-learn==1.0.2',
        'imageio==2.19.3',
        'xgboost==1.6.1',
        'notebook==6.4.12',
        'opencv-python==4.6.0.66',
    ],
    python_requires="==3.7.13",
    setup_requires=['cffi>=1.15.1'],
    cffi_modules=[
        "airxd/builder.py:ffibuilder"
    ]
)
