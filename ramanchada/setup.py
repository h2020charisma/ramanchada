#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fs:
    long_description = fs.read()

setup(
    name="ramanchada",
    version="0.0.1",
    author="Bastian Barton",
    author_email="bastian.Barton@lbf.fraunhofer.de",
    description="Raman spectra standardisation H2020 CHARISMA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.cc-asp.fraunhofer.de/",
    project_urls={
        "Issue tracker": "tbd",
        "Project" : "https://www.h2020charisma.eu/project-overview",
        "Source" : "https://gitlab.cc-asp.fraunhofer.de/"
    },
    classifiers=[
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="Raman spectra standardisation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="~=3.5",
    install_requires=[
        "pandas",
        "scipy"
    ]
)
