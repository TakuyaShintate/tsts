from codecs import open
from os import path
from typing import Any, List

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup


def choose_requirement(main: Any, secondary: Any) -> Any:
    try:
        get_distribution(main)
    except DistributionNotFound:
        return secondary
    return main


def get_install_requirements(
    install_requires: List[str],
    choose_install_requires: List[Any],
) -> List[Any]:
    for main, secondary in choose_install_requires:
        install_requires.append(choose_requirement(main, secondary))
    return install_requires


def install_requires() -> List[str]:
    requires = [
        "numpy",
        "torch",
        "tqdm",
        "yacs",
        "numba",
        "seaborn",
        "terminaltables",
    ]
    return requires


here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="tsts",
    packages=[p for p in find_packages() if p.startswith("tsts")],
    license="MIT License",
    install_requires=get_install_requirements(
        install_requires(),
        [],
    ),
    author="Takuya Shintate",
    author_email="kmdbn2hs@gmail.com",
    url="https://github.com/TakuyaShintate/tsts",
    description="toolset for time series forecasting",
    version="0.8.3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="tsts",
    include_package_data=True,
)
