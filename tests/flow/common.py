from VecSim import *
import numpy as np
from scipy import spatial
from  numpy.testing import assert_allclose
import time
import os, sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
READIES = os.path.join(ROOT, "deps/readies")
sys.path.insert(0, READIES)
import paella
