import os
from setuptools import setup, find_packages

version_filename = os.path.join(os.path.split(__file__)[0], "clip/version.py")
with open(version_filename) as version_file:
    exec(compile(version_file.read(), version_filename, "exec"))

setup(
    name='clip',
    version=version_from_git(),
    author="Jason O'Kane",
    author_email='jokane@tamu.edu',
    description='A package for creating video clips.',
    packages=find_packages(),    
    install_requires=['opencv-python>=4.6.0.66',
                      'numba>=0.56.4',
                      'pdf2image>=1.16.0',
                      'progressbar2>=4.2.0',
                      'scipy>=1.9.3',
                      'soundfile>=0.11.0',
                      'rosbags>=0.10.4']
)
