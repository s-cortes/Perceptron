import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read( )

setuptools.setup(
    name="s-cortes",
    version="0.0.1",
    author="Santiago Cortes Fernandez",
    author_email="santiago.cortes0{at}gmail.com",
    description="Simple Rosenblatt's Perceptron implementation for distinguishing a pair of linear of separable patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/s-cortes/perceptron",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)