/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../operator_pool.h"

namespace cudaq {

// adapted from
// https://github.com/mayhallgroup/adapt-vqe/blob/master/src/operator_pools.py

class spin_complement_gsd
    : public details::operator_pool_impl<spin_complement_gsd> {

public:
  std::vector<spin_op> generate(
      const std::unordered_map<std::string, std::any> &config) const override;
  CUDAQ_REGISTER_OPERATOR_POOL(spin_complement_gsd)
};
} // namespace cudaq