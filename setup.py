from setuptools import setup

setup(
    name='bjlib',
    url='https://github.com/jladan/package_demo',
    author='Baptiste Jost',
    author_email='jost@apc.in2p3.fr',
    packages=['bjlib'],
    install_requires=['numpy', 'healpy', 'pysm', 'copy', 'pymaster', 'IPython',
                      'matplotlib', 'time', 'astropy', 'scipy', 'mpl_toolkits',
                      'pandas', 'fgbuster', 'V3calc'],
    version='0.0.1',
    license='MIT',
    description='My personal library',
)
