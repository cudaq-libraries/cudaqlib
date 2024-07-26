/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "fermion_operators.h"
#include "fermion_compiler.h"

namespace cudaq::operators {
spin_op adag(std::size_t numQubits, std::size_t p,
             const std::string transform) {
  auto transformer = fermion_compiler::get(transform);
  return transformer->adag(numQubits, p);
}
spin_op a(std::size_t numQubits, std::size_t q, const std::string transform) {
  auto transformer = fermion_compiler::get(transform);
  return transformer->adag(numQubits, q);
}

} // namespace cudaq::operators