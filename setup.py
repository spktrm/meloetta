from setuptools import find_packages, setup

from meloetta import __version__

requirements = [
    "py_mini_racer==0.6.0",
    "requests==2.28.1",
    "websockets==10.4",
]


pretrain_requirements = [
    "transformers[torch]==4.18.0",
    "sentence-transformers==2.2.0",
    "huggingface-hub==0.4.0",
]


setup(
    name="meloetta",
    version=__version__,
    url="https://github.com/spktrm/meloetta",
    author="Joseph Twin",
    author_email="joseph.twin14@gmail.com",
    long_description=open("README.md").read(),
    packages=find_packages(),
    setup_requires=["wheel"],
    install_requires=requirements,
    extras_require=dict(
        core_tests=pretrain_requirements,
    ),
)
