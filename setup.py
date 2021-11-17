
from setuptools import setup, find_packages

setup(
    name="bbhnet-inference",
    version="0.0.1",
    author="Tri Nguyen",
    author_email="tri.vt.nguyen@gmail.com",
    license='LICENSE',
    description='Inference code for BBH network',
    long_description=open('README.md').read(),
    packages=find_packages(),
    classifiers=(
      'Programming Language :: Python',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Intended Audience :: End Users/Desktop',
      'Intended Audience :: Developers',
      'Natural Language :: English',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Astronomy',
      'Topic :: Scientific/Engineering :: Physics',
      'Operating System :: POSIX',
      'Operating System :: Unix',
      'Operating System :: MacOS',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ),
    python_requires='>=3.6',
)


