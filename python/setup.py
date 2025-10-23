# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for pylightnet."""

import os
import shutil
import subprocess
import sys

from setuptools import Command, find_packages, setup

try:
    from setuptools.command.build import build
except ImportError as e:
    print(
        "Error: Failed to import 'build' from setuptools.command.build\n"
        "\n"
        "This package requires setuptools>=80.9.0,<81.0.0\n"
        "\n"
        "Please upgrade setuptools to the required version:\n"
        "  pip install 'setuptools>=80.9.0,<81.0.0'\n"
        "\n"
        f"Original error: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

rebuild_trtlightnet = os.environ.get("REBUILD_TRTLIGHTNET", "1") == "1"


class CustomBuild(build):
    description = "Custom build command"

    def run(self):
        self.run_command("build_lightnet_infer")
        super().run()


class BuildLightnetInfer(Command):
    """Custom command to build liblightnetinfer.so using CMake.

    Reuses libcnpy.so from CMake build instead of building separately.
    """

    description = "Build liblightnetinfer.so using CMake"
    user_options = []

    def initialize_options(self):
        """Initialize options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def _find_libcnpy(self, build_dir):
        """Locate libcnpy.so from CMake build or system installation.

        Args:
            build_dir (str): Path to CMake build directory

        Returns:
            str: Absolute path to libcnpy.so

        Raises:
            FileNotFoundError: If libcnpy.so not found in any search path
        """
        search_paths = [
            # FetchContent build directory
            os.path.join(build_dir, "_deps", "cnpy-build", "libcnpy.so"),
            # System installation paths
            "/usr/local/lib/libcnpy.so",
            "/usr/lib/x86_64-linux-gnu/libcnpy.so",
        ]

        for path in search_paths:
            if os.path.exists(path):
                print(f"Found libcnpy.so at: {path}")
                return path

        # If not found, raise detailed error
        error_msg = (
            "Error: libcnpy.so not found in any of the following paths:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
            + "\n\nTo resolve this issue, try one of the following:\n"
            + "1. Install cnpy system-wide:\n"
            + "   git clone https://github.com/rogersce/cnpy.git\n"
            + "   cd cnpy && mkdir build && cd build\n"
            + "   cmake .. && make && sudo make install\n"
            + "2. Ensure CMakeLists.txt FetchContent is enabled (FETCH_CNPY_IF_MISSING=ON)\n"
            + f"3. Check CMake build logs in {build_dir} for errors\n"
        )
        raise FileNotFoundError(error_msg)

    def run(self):
        """Build lightnetinfer library and copy required .so files to package."""
        setup_root_dir = os.path.dirname(os.path.realpath(__file__))
        pylightnet_dir = os.path.join(setup_root_dir, "pylightnet")
        lightnet_dir = os.path.dirname(setup_root_dir)
        lightnet_build_dir = os.path.join(lightnet_dir, "build")

        try:
            # Handle build directory based on REBUILD_TRTLIGHTNET
            if rebuild_trtlightnet:
                # Clean build: remove existing build directory
                if os.path.exists(lightnet_build_dir):
                    print("Cleaning build directory (REBUILD_TRTLIGHTNET=1)")
                    shutil.rmtree(lightnet_build_dir)
                os.makedirs(lightnet_build_dir, exist_ok=True)

                # Build lightnetinfer using CMake
                subprocess.check_call(["cmake", ".."], cwd=lightnet_build_dir)
                subprocess.check_call(["make", "-j"], cwd=lightnet_build_dir)
            else:
                # Incremental build: preserve existing build directory
                print("Skipping build directory clean (REBUILD_TRTLIGHTNET=0)")

                if os.path.exists(lightnet_build_dir):
                    # Build directory exists, skip CMake build and use existing artifacts
                    print(f"Reusing existing build artifacts from {lightnet_build_dir}")
                else:
                    # Build directory doesn't exist, create and build
                    os.makedirs(lightnet_build_dir, exist_ok=True)
                    subprocess.check_call(["cmake", ".."], cwd=lightnet_build_dir)
                    subprocess.check_call(["make", "-j"], cwd=lightnet_build_dir)

            # Find and copy libcnpy.so
            libcnpy_path = self._find_libcnpy(lightnet_build_dir)
            shutil.copy(libcnpy_path, os.path.join(pylightnet_dir, "libcnpy.so"))
            print(f"Successfully copied libcnpy.so from {libcnpy_path}")

            # Copy liblightnetinfer.so to package directory
            shutil.copy(
                os.path.join(lightnet_build_dir, "liblightnetinfer.so"),
                os.path.join(pylightnet_dir, "liblightnetinfer.so"),
            )
            print("Successfully built liblightnetinfer.so")

            # Preserve build directory for future incremental builds
            print(f"Build directory preserved at: {lightnet_build_dir}")

        except subprocess.CalledProcessError as e:
            print(f"Failed to execute command: {e}")
            raise
        except FileNotFoundError as e:
            print(str(e))
            raise


setup(
    name="pylightnet",
    version="1.0.0",
    packages=find_packages(),
    package_data={
        "": [
            "liblightnetinfer.so",
            "libcnpy.so",
        ],
    },
    install_requires=[
        "opencv-python",
        "numpy<2.0",
    ],
    setup_requires=[
        "setuptools>=80.9.0,<81.0.0",
    ],
    extras_require={
        "dev": [
            "ruff",
            "pytest",
        ],
    },
    cmdclass={
        "build": CustomBuild,
        "build_lightnet_infer": BuildLightnetInfer,
    },
)
