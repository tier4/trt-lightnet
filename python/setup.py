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

from setuptools import Command, find_packages, setup
from setuptools.command.build import build

skip_ext = os.environ.get("SKIP_EXT", "0") == "1"


class CustomBuild(build):
    description = "Custom build command"

    def run(self):
        if not skip_ext:
            self.run_command("build_lightnet_infer")
            build.run(self)
        else:
            print("SKIP_EXT=1 detected: skipping build_lightnet_infer")


class BuildLightnetInfer(Command):
    """Custom command to execute Makefile steps for building liblightnetinfer.so."""

    description = "Build liblightnetinfer.so using Makefile steps"
    user_options = []

    def initialize_options(self):
        """Initialize options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def _clone_cnpy(self, cnpy_dir):
        """Clone cnpy repository.

        Args:
            cnpy_dir (str): Path to clone cnpy repository.

        """
        try:
            subprocess.check_call(
                ["git", "clone", "https://github.com/rogersce/cnpy.git", cnpy_dir],
            )
        except subprocess.CalledProcessError as e:
            print(f"Error cloning cnpy: {e}")
            raise

    def run(self):
        """Build liblightnetinfer.so."""
        setup_root_dir = os.path.dirname(os.path.realpath(__file__))
        pylightnet_dir = os.path.join(setup_root_dir, "pylightnet")
        cnpy_dir = os.path.join(setup_root_dir, "cnpy")
        if os.path.exists(cnpy_dir):
            shutil.rmtree(cnpy_dir)
        self._clone_cnpy(cnpy_dir)

        try:
            cnpy_build_dir = os.path.join(cnpy_dir, "build")
            os.makedirs(cnpy_build_dir)

            # Run cmake and make within cnpy/build
            subprocess.check_call(
                [
                    "cmake",
                    "..",
                    f"-DCMAKE_INSTALL_PREFIX={os.path.abspath(cnpy_build_dir)}",
                ],
                cwd=cnpy_build_dir,
            )
            subprocess.check_call(["make", "-j"], cwd=cnpy_build_dir)
            subprocess.check_call(["make", "install"], cwd=cnpy_build_dir)

            # Copy liblightnetinfer.so to package directory
            shutil.copy(
                os.path.join(cnpy_build_dir, "libcnpy.so"),
                os.path.join(pylightnet_dir, "libcnpy.so"),
            )

            print("Successfully built libcupy.so")

            # Clean build directory
            lightnet_dir = os.path.dirname(setup_root_dir)
            lightnet_build_dir = os.path.join(lightnet_dir, "build")
            if os.path.exists(lightnet_build_dir):
                shutil.rmtree(lightnet_build_dir)
            os.makedirs(lightnet_build_dir, exist_ok=True)

            # Set environment variables and make liblightnetinfer.so
            os.environ["CPLUS_INCLUDE_PATH"] = (
                f"{os.path.join(cnpy_build_dir, 'include')}:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
            )
            os.environ["LIBRARY_PATH"] = (
                f"{os.path.join(cnpy_build_dir, 'lib')}:{os.environ.get('LIBRARY_PATH', '')}"
            )
            subprocess.check_call(["cmake", ".."], cwd=lightnet_build_dir)
            subprocess.check_call(["make", "-j"], cwd=lightnet_build_dir)

            # Copy liblightnetinfer.so to package directory
            shutil.copy(
                os.path.join(lightnet_build_dir, "liblightnetinfer.so"),
                os.path.join(pylightnet_dir, "liblightnetinfer.so"),
            )

            print("Successfully built liblightnetinfer.so")

            # Clean build directory
            shutil.rmtree(lightnet_build_dir)
            shutil.rmtree(cnpy_dir)

        except subprocess.CalledProcessError as e:
            print(f"Failed to execute command: {e}")
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
        "numpy",
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
