from setuptools import setup, find_packages

setup(
    name='sonicprep',
    version='0.0.0-a1',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.5',
        'librosa>=0.10.1',
    ],
)
