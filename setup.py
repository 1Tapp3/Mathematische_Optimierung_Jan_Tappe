from setuptools import setup, find_packages

print("hallo2")
setup(
    name='OptimizationDemoCode',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        numpy, cvxopt
    ],
    test_suite='tests',
)
