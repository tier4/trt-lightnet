# pylightnet
**pylightnet** is a Python wrapper library for [trt-lightnet](https://github.com/hdwlab/trt-lightnet).

## Operating Environment
Please refer to [this page](https://github.com/hdwlab/trt-lightnet) for the operating environment.

## Prerequisites
Please ensure setuptools is updated to the latest version before installation. This package has been tested with setuptools version 80.9.0.

```bash
$ pip install 'setuptools>=80.0.0,<81.0.0'
```

## Installation Method
You can install pylightnet using the following command:

```bash
$ pip install .
```

## Run demo script
After placing the configuration file(`flagfile`) and its associated model files, you can execute the demo script using the following command:

```bash
$ python scripts/demo.py \
    -f <path/to/flagfile.txt> \
    -v <path/to/video.mp4>
```

## Installation & Test with Docker
You can install pylightnet and test it with Docker using the following command:
```bash
$ make test-pylightnet
```
