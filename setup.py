from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="topolosses",
    version="0.0.10",
    description="A collection of losses and metrices for topology preserving image segmentation.",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Janek Falkenstein",
    author_email="j.falkenstein@tum.de",
    license="MIT",
    classifiers=[
        "Operating System :: OS Independent",
        # TODO add more classifiers
    ],
    # TODO find out what the dependencies are 

    # all the packages needed for this project,
    install_requires=[],
    extras_require={},
    # TODO how x
    python_requires=">=3.7",
)
