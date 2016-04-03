import os
import sys
from setuptools import setup, find_packages


pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(pwd)

try:
    README = open(os.path.join(pwd, 'docs/pypi.rst')).read()
except IOError:
    README = ''

try:
    import sknn
    VERSION = sknn.__version__
except ImportError:
    VERSION = 'N/A'


install_requires = [
    'scikit-learn>=0.17',
    'Theano>=0.8',
    'Lasagne>=0.1',
    'colorama' if sys.platform == 'win32' else '',
]

tests_require = [
    'nosetests',
]

docs_require = [
    'Sphinx',
]

setup(name='scikit-neuralnetwork',
      version=VERSION,
      description="Deep neural networks without the learning cliff! A wrapper library compatible with scikit-learn.",
      long_description=README,
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          ],
      keywords='deep learning, neural networks',
      url='https://github.com/aigamedev/scikit-neuralnetwork',
      license='BSD 3-clause license',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      extras_require={
          'testing': tests_require,
          'docs': docs_require,
          },
      )
