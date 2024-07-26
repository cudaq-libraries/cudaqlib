/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/spin_op.h"
#include "cudaqlib/utils/extension_point.h"
#include "fermion_op.h"

namespace cudaq::operators {

/// @brief The `fermion_compiler` type serves as a base class defining
/// and interface for clients to map `fermion_op` instances to
/// `cudaq::spin_op` instances.
class fermion_compiler : public extension_point<fermion_compiler> {
public:
  /// @brief Given a fermionic representation of an operator
  /// generate an equivalent operator on spins.
  virtual spin_op generate(const fermion_op &fermionOp) = 0;
  virtual spin_op adag(std::size_t numQubits, std::size_t p) = 0;
  virtual spin_op a(std::size_t numQubits, std::size_t q) = 0;
  virtual ~fermion_compiler() {}
};

CUDAQ_DEFINE_EXTENSION_IMPL(fermion_compiler)

#define CUDAQ_REGISTER_FERMION_TO_SPIN(TYPE)                                   \
  static inline const std::string class_identifier = #TYPE;                    \
  static std::unique_ptr<fermion_compiler> create() {                          \
    return std::make_unique<TYPE>();                                           \
  }

} // namespace cudaq::operators