#!/usr/bin/env python3

import sys
import os
import argparse

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
READIES = os.path.join(ROOT, "deps/readies")
sys.path.insert(0, READIES)
import paella

#----------------------------------------------------------------------------------------------

class VecSimSetup(paella.Setup):
    def __init__(self, nop=False):
        paella.Setup.__init__(self, nop)

    def common_first(self):
        self.install_downloaders()
        self.pip_install("wheel")
        self.pip_install("setuptools --upgrade")

        self.install("git")

        self.run("%s/bin/enable-utf8" % READIES)
        self.run("%s/bin/getcmake" % READIES)

    def debian_compat(self):
        self.run("%s/bin/getgcc --modern" % READIES)
        self.install("python3-dev clang-format valgrind")

    def redhat_compat(self):
        self.install("redhat-lsb-core")
        self.run("%s/bin/getgcc --modern" % READIES)

    def fedora(self):
        self.run("%s/bin/getgcc --modern" % READIES)

    def macos(self):
        self.install_gnu_utils()
        self.run("%s/bin/getgcc --modern" % READIES)

    def common_last(self):
        self.run("python3 %s/bin/getrmpytools" % READIES)
        self.run("%s/bin/getclang" % READIES)

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

VecSimSetup(nop = args.nop).setup()
