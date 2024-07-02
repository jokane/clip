from setuptools import setup, find_packages

setup(
    name='clip',
    version='0.4.0',
    author="Jason O'Kane",
    author_email='jokane@tamu.edu',
    description='A package for creating video clips.',
    packages=find_packages(),    
    install_requires=[ 'opencv-python>=4.6.0.66', 'numba>=0.56.4', 'pdf2image>=1.16.0', 'progressbar2>=4.2.0', 'scipy>=1.9.3', 'soundfile>=0.11.0' ]
)
