""" boostIV """

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="boostIV",
    version="0.0.1",
    author="Edvard Bakhitov",
    author_email="bakhitov@sas.upenn.edu",
    description="Implementation of boostIV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(where='/Users/Ed/Documents/Penn/MLNPIV/boostIV'),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'sklearn', 'numdifftools'],
    package_dir={'': '/Users/Ed/Documents/Penn/MLNPIV/boostIV'}
)
