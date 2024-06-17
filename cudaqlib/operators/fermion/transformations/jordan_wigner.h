/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "../fermion_to_spin.h"

namespace cudaq::operators {

/// @brief Map fermionic operators to spin operators via the
/// Jordan-Wigner transformation.
class jordan_wigner : public details::fermion_to_spin_impl<jordan_wigner> {
public:
  spin_op generate(const fermion_op &fermionOp) override;
  CUDAQ_REGISTER_FERMION_TO_SPIN(jordan_wigner)
};
} // namespace cudaq::operators