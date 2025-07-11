"""Setup script for the project."""

import re

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("iluvattnshun/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()


with open("iluvattnshun/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in iluvattnshun/__init__.py"
version: str = version_re.group(1)


setup(
    name="iluvattnshun",
    version=version,
    description="A framework for quick attention-based experiments.",
    author="michael-lutz",
    url="https://github.com/michael-lutz/iluvattnshun",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    packages=find_packages(),
)
