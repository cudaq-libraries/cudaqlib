/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "../fermion_compiler.h"

namespace cudaq::operators {

/// @brief Map fermionic operators to spin operators via the
/// Jordan-Wigner transformation.
class jordan_wigner : public details::fermion_compiler_impl<jordan_wigner> {
public:
  spin_op generate(const fermion_op &fermionOp) override;
  spin_op adag(std::size_t numQubits, std::size_t p) override;
  spin_op a(std::size_t numQubits, std::size_t q) override;
  CUDAQ_REGISTER_FERMION_TO_SPIN(jordan_wigner)
};
} // namespace cudaq::operators