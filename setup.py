from setuptools import setup

# Boilerplate for integrating with PyTest
from setuptools.command.test import test
import sys


class PyTest(test):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]
    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

# The actual setup metadata
setup(
    name='tensorspace',
    version='0.0.1',
    description='A reference implementation of an AI lab.',
    long_description=open("README.rst").read(),
    keywords='machine_learning artificial_intelligence devops',
    author='JJ Ben-Joseph',
    author_email='opensource@phrostbyte.com',
    python_requires='>=3.6',
    url='https://www.phrostbyte.com/',
    license='Apache',
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent'
    ],
    py_modules=['tensorspace'],
    install_requires=['tqdm', 'sqlalchemy', 'python-dateutil', 'annoy', 
                      'tensorflow_hub', 'scikit-image'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tensorspace=tensorspace.__main__',
        ],
    },
    cmdclass = {'test': PyTest}
)