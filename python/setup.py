import os
import shutil
import subprocess
from distutils.cmd import Command

from setuptools import find_packages, setup
from setuptools.command.build import build


class CustomBuild(build):
    description = "Custom build command"

    def run(self):
        self.run_command("build_lightnet_infer")
        build.run(self)


class BuildLightnetInfer(Command):
    """
    Custom command to execute Makefile steps for building liblightnetinfer.so.
    """

    description = "Build liblightnetinfer.so using Makefile steps"
    user_options = []

    def initialize_options(self):
        """Initialize options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def _clone_cnpy(self):
        """Clone cnpy repository."""
        try:
            subprocess.check_call(
                "git clone https://github.com/rogersce/cnpy.git", shell=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error cloning cnpy: {e}")
            raise

    def run(self):
        """Build liblightnetinfer.so."""
        # Execute make steps
        script_dir = os.path.dirname(os.path.realpath(__file__))
        cnpy_dir = os.path.join(script_dir, "lib", "cnpy")
        if not os.path.exists(cnpy_dir):
            lib_dir = os.path.join(script_dir, "lib")
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            os.chdir(lib_dir)
            self._clone_cnpy()
        try:
            cnpy_build_dir = os.path.join(cnpy_dir, "build")
            if os.path.exists(cnpy_build_dir):
                shutil.rmtree(cnpy_build_dir)
            os.makedirs(cnpy_build_dir)

            # Run cmake and make within cnpy/build
            os.chdir(cnpy_build_dir)
            subprocess.check_call(
                [
                    "cmake",
                    "..",
                    f"-DCMAKE_INSTALL_PREFIX={os.path.abspath(cnpy_build_dir)}",
                ]
            )
            subprocess.check_call(["make", "-j"])
            subprocess.check_call(["make", "install"])

            # Clean build directory
            lightnet_dir = os.path.dirname(script_dir)
            lightnet_build_dir = os.path.join(lightnet_dir, "build")
            os.chdir(lightnet_dir)
            if os.path.exists(lightnet_build_dir):
                shutil.rmtree(lightnet_build_dir)
            os.makedirs(lightnet_build_dir, exist_ok=True)
            os.chdir(lightnet_build_dir)

            # Set environment variables and make liblightnetinfer.so
            os.environ["CPLUS_INCLUDE_PATH"] = (
                f"{os.path.join(cnpy_build_dir, 'include')}:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
            )
            os.environ["LIBRARY_PATH"] = (
                f"{os.path.join(cnpy_build_dir, 'lib')}:{os.environ.get('LIBRARY_PATH', '')}"
            )
            subprocess.check_call(["cmake", ".."])
            subprocess.check_call(["make", "-j"])

            # Copy liblightnetinfer.so to package directory
            shutil.copy(
                os.path.join(lightnet_build_dir, "liblightnetinfer.so"),
                os.path.join(script_dir, "liblightnetinfer.so"),
            )

            print("Successfully built liblightnetinfer.so")
            os.chdir(script_dir)

        except subprocess.CalledProcessError as e:
            print(f"Failed to execute command: {e}")
            raise


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
    cmdclass={
        "build": CustomBuild,
        "build_lightnet_infer": BuildLightnetInfer,
    },
)
