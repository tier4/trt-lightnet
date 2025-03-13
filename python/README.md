# pylightnet
**pylightnet** is a Python wrapper library for [trt-lightnet](https://github.com/hdwlab/trt-lightnet).

## Operating Environment
Please refer to [this page](https://github.com/hdwlab/trt-lightnet) for the operating environment.

## Installation Method
You can install pylightnet using the following command:

```bash
$ pip install -U setuptools pip
$ pip install .
```

## Run demo script
After placing the configuration file(`flagfile`) and its associated model files, you can execute the demo script using the following command:

```bash
$ python scripts/demo.py \
    -f <path/to/flagfile.txt> \
    -v <path/to/video.mp4>
```
