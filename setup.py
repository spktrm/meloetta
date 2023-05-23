from setuptools import find_packages, setup

from meloetta import __version__

requirements = [
    "beautifulsoup4==4.12.2",
    "jax==0.3.14",
    "detoxify==0.5.1",
    "pandas==1.3.5",
    "numpy==1.21.5",
    "py_mini_racer==0.6.0",
    "plotly==5.11.0",
    "scikit_learn==1.2.2",
    "requests==2.26.0",
    "torch==1.13.0",
    "setuptools==58.0.0",
    "tqdm==4.64.1",
    "websockets==10.1",
]


pretrain_requirements = [
    "transformers[torch]==4.18.0",
    "sentence-transformers==2.2.0",
    "huggingface-hub==0.4.0",
]


long_description = """Meloetta is a Pokémon Battle Client for interacting with Pokémon Showdown written in Python. This project was born out of frustration for currently existing tools and their lack of dependency on Zarel's (PS Creator) existing code for handling client server interation.

The client works by reading messages from an asyncio stream and forwarding these to the javascript client battle object with PyMiniRacer. This concept was taken from [metagrok](https://github.com/yuzeh/metagrok).

As is, the necessary javascript source files come with the pip install. Whenever Pokemon Showdown Client releases an update, the client code can also be automatically updated from the source."""

setup(
    name="meloetta",
    version=__version__,
    url="https://github.com/spktrm/meloetta",
    author="Joseph Twin",
    author_email="joseph.twin14@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(),
    install_requires=requirements,
    extras_require=dict(
        core_tests=pretrain_requirements,
    ),
)
