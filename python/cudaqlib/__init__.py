# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import os, os.path
from ._pycudaqlib import *
from .kernels import *
try:
    from .algorithms.gqe import gqe
except:
    print('[cudaqlib warning] Could not import GQE module. Install required modules (e.g. torch)')
    pass