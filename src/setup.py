from setuptools import setup, find_packages

setup(
    name='dqnroute',
    version='0.1',
    description='Simple routing simulator in Python',
    author='Ivan Panchishin',
    author_email='rot1tweiler@gmail.com',
    packages=find_packages(),
    install_requires=(
        'networkx>=2.3',
        'simpy>=3.0.11',
        'numpy>=1.15.3',
        'pyyaml>=4.2b1',
        'torch>=1.0.1'
    )
)
