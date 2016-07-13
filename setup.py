""" Setup script for sox. """
from setuptools import setup

if __name__ == "__main__":
    setup(
        name='sox',

        version='1.1.3',

        description='Python wrapper around SoX.',

        author='Rachel Bittner',

        author_email='rachel.bittner@nyu.edu',

        url='https://github.com/rabitt/pysox',

        download_url='http://github.com/rabitt/pysox/releases',

        packages=['sox'],

        package_data={'sox': []},

        long_description="""Python wrapper around SoX.""",

        keywords='audio effects SoX',

        license='MIT',

        install_requires=[
        ],

        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        }
    )
