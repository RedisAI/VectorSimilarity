import os
import re
import subprocess
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = HERE
READIES = os.path.join(ROOT, "deps/readies")
sys.path.insert(0, READIES)
import paella


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "DEBUG" if debug else "RelWithDebInfo"

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-Wno-dev",
            "--no-warn-unused-cli",
            "-DBUILD_TESTS=OFF",
        ]
        build_args = []

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


setup(
    name="VecSim",
    version="0.0.1",
    author="Redis, Inc. CTO Team",
    author_email="oss@redis.com",
    description="Python library around collection of vector similarity algorithm",
    long_description="",
    ext_modules=[CMakeExtension("VecSim", "src/python_bindings")],
    cmdclass={"build_ext": CMakeBuild}
)
