#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for miniManus.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="minimanus",
    version="0.1.0",
    author="miniManus Team",
    author_email="info@minimanus.app",
    description="A mobile-focused framework that runs on Linux in Termux for Android phones",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/miniManus",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Topic :: Communications :: Chat",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "minimanus=minimanus.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "minimanus": ["data/*", "config/*"],
    },
)
