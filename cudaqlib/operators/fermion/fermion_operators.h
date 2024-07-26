/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/spin_op.h"

namespace cudaq::operators {
spin_op adag(std::size_t numQubits, std::size_t p,
             const std::string transform = "jordan_wigner");
spin_op a(std::size_t numQubits, std::size_t q,
          const std::string transform = "jordan_wigner");
} // namespace cudaq::operators