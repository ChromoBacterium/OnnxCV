import os
from setuptools import setup, find_packages

with open("requirements.txt", "r") as req:
    req = req.read().splitlines()

with open("README.md", "r") as rm:
    rm = rm.read()

setup(
    name='onnxcv',
    version='0.0.1',
    author="Ayaz Amin",
    packages=find_packages('onnxcv'),
    install_requires=req,
    description="An ONNX inference engine for computer vision"
    long_description=rm,
    url="https://github.com/ChromoBacterium/OnnxCV"
)