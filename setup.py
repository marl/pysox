""" Setup script for sox. """
from setuptools import setup

import imp

version = imp.load_source('sox.version', 'sox/version.py')

if __name__ == "__main__":
    setup(
        name='sox',
        version=version.version,
        description='Python wrapper around SoX.',
        author='Rachel Bittner',
        author_email='rachel.bittner@nyu.edu',
        url='https://github.com/rabitt/pysox',
        download_url='http://github.com/rabitt/pysox/releases',
        packages=['sox'],
        package_data={'sox': []},
        long_description="""Python wrapper around SoX.""",
        keywords='audio effects SoX',
        license='BSD-3-Clause',
        install_requires=[
            'numpy >= 1.9.0',
        ],
        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'pysoundfile >= 0.9.0',
            ],
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        }
    )
