import setuptools

with open("README.txt", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "SunSpot", 
    version = "0.0.1",
    author = "Sophie Mathieu",
    author_email = "sph.mathieu@gmail.com",
    description = "Non parametric robust monitoring of sunspot data",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/sophiano/SunSpot",
    packages = setuptools.find_packages(where='SunSpot'),
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    data_files=[('Ns', ['data/Ns'])],
)