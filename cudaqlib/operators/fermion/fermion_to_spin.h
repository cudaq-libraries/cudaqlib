/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaqlib/utils/extension_point.h"
#include "cudaq/spin_op.h"
#include "fermion_op.h"

namespace cudaq::operators {

/// @brief The `fermion_to_spin` type serves as a base class defining
/// and interface for clients to map `fermion_op` instances to
/// `cudaq::spin_op` instances.
class fermion_to_spin : public extension_point<fermion_to_spin> {
public:
  /// @brief Given a fermionic representation of an operator
  /// generate an equivalent operator on spins.
  virtual spin_op generate(const fermion_op &fermionOp) = 0;
};

CUDAQ_DEFINE_EXTENSION_IMPL(fermion_to_spin)

#define CUDAQ_REGISTER_FERMION_TO_SPIN(TYPE)                                  \
  static inline const std::string class_identifier = #TYPE;                    \
  static std::unique_ptr<fermion_to_spin> create() {                           \
    return std::make_unique<TYPE>();                                           \
  }

} // namespace cudaq::operators