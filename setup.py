from setuptools import setup, find_packages
from codecs import open
from os import path

# Marc's steps for uploading new verison
# 1. change version number below
# 2. python setup.py sdist
# 3. python setup.py sdist upload

here = path.abspath(path.dirname(__file__))

setup(
    name='pysdr',
    version='1.1', # lets go with a simple 2 digit versioning system starting at 1.0
    description='A software-defined radio (SDR) framework',
    long_description='A software-defined radio (SDR) framework designed for extremely rapid development of computationally intensive SDR apps with beautiful GUIs',
    url='https://github.com/pysdr/pysdr',
    author='Marc L',
    author_email='777arc@vt.edu',
    license='GPL-3.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project?
        'Development Status :: 1 - Planning',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering', # closest thing i could find =/

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        #'Programming Language :: Python :: 2',
        #'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',  # havent tested it yet with  python3
        #'Programming Language :: Python :: 3.3',
        #'Programming Language :: Python :: 3.4',
        #'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='software-defind radio dsp sdr digital signal processing wireless communications',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy',
        'pyrtlsdr',
        'scipy',
        'pyqt5',
        'pyqtgraph'
]
)
