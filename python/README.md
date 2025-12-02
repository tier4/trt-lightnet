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

### Build Configuration

The build process can be controlled using the `REBUILD_TRTLIGHTNET` environment variable:

#### Full Rebuild (Default)
By default, or when `REBUILD_TRTLIGHTNET=1` is set, the build directory is cleaned before building. This ensures a clean build from scratch:

```bash
# Default behavior (full rebuild)
$ pip install .

# Explicit full rebuild
$ REBUILD_TRTLIGHTNET=1 pip install .
```

#### Incremental Build
When `REBUILD_TRTLIGHTNET=0` is set, the build directory is preserved between builds. If the build directory already exists, the CMake build process is skipped and existing build artifacts are reused, significantly reducing build time:

```bash
# Incremental build (skips CMake build if build directory exists)
$ REBUILD_TRTLIGHTNET=0 pip install .
```

**When to use each option:**
- **Full rebuild (`REBUILD_TRTLIGHTNET=1` or default)**:
  - First time installation
  - After updating C++/CUDA source code
  - When you want to ensure a clean build
  - CI/CD pipelines

- **Incremental build (`REBUILD_TRTLIGHTNET=0`)**:
  - Rapid development iterations with Python code changes only
  - When debugging and testing frequently
  - After the initial build is complete

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
