/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "gse/py_gse.h"
#include "operators/py_operators.h"
#include "optim/py_optimizer.h"
#include "qec/py_qec.h"

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(_pycudaqlib, mod) {
  mod.doc() = "Python bindings for the CUDAQ-X Libraries.";
  cudaq::optim::bindOptim(mod);
  cudaq::gse::bindGse(mod);
  cudaq::operators::bindOperators(mod);
  cudaq::qec::bindQec(mod);
}
