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
    ]
    return requires


setup(
    name="tsts",
    packages=[p for p in find_packages() if p.startswith("tsts")],
    install_requires=get_install_requirements(
        install_requires(),
        [],
    ),
    version="0.0.1",
    include_package_data=True,
)
