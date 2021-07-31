import os

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

CHOOSE_INSTALL_REQUIRES = []


def choose_requirement(main, secondary):
    try:
        get_distribution(main)
    except DistributionNotFound:
        return secondary
    return main


def get_install_requirements(install_requires, choose_install_requires):
    for main, secondary in choose_install_requires:
        install_requires.append(choose_requirement(main, secondary))
    return install_requires


def install_requires():
    requires = [
        "numpy",
        "torch",
        "tqdm",
    ]
    requires += models_requires()
    return requires


def models_requires():
    requires = ["scikit-image"]
    return requires


setup(
    name="tsts",
    packages=[p for p in find_packages() if p.startswith("tsts")],
    install_requires=get_install_requirements(
        install_requires(), CHOOSE_INSTALL_REQUIRES
    ),
    version="0.0.1",
    include_package_data=True,
)
