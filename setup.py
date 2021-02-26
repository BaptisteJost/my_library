from setuptools import setup

setup(
    name='bjlib',
    url='https://github.com/jladan/package_demo',
    author='Baptiste Jost',
    author_email='jost@apc.in2p3.fr',
    packages=['bjlib'],
    install_requires=['numpy', 'healpy', 'pysm'],
    version='0.0.2.16.4',
    license='MIT',
    description='My personal library',
)
