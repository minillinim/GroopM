from setuptools import setup 

setup(
    name='GroopM',
    version='0.3.5',
    author='Michael Imelfort',
    author_email='mike@mikeimelfort.com',
    packages=['groopm'],
    scripts=['bin/groopm'],
	url='http://pypi.python.org/pypi/GroopM/',
    license='LICENSE.txt',
    description='Metagenomic binning suite',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.6.1",
        "scipy >= 0.15.0",
        "matplotlib >= 1.1.0",
        "tables >= 2.3"
    ],
)
