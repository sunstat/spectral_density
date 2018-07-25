import setuptools
setuptools.setup(
    name="spectral_density",
    version="0.0.1",
    author="Yiming Sun",
    author_email="sunstat@stanford.edu",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'tensorly',
        'matplotlib',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
