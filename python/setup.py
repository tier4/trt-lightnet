from setuptools import find_packages, setup

setup(
    name="pylightnet",
    version="1.0.0",
    packages=find_packages(),
    package_data={
        "": ["liblightnetinfer.so"],
    },
    install_requires=[
        "opencv-python",
        "numpy",
    ],
)
