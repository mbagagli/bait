from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()

setup(
    name="bait",
    version="2.5.9",
    author="Matteo Bagagli",
    author_email="matteo.bagagli@ingv.it",
    description="an ITerative BAer picker algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mbagagli/bait",
    python_requires='>=3.6',
    setup_requires=['wheel'],
    install_requires=required_list,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
    ],
)
