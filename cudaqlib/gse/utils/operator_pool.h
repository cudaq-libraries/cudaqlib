/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <any>
#include <optional>
#include <unordered_map>

#include "cudaq/spin_op.h"
#include "cudaqlib/utils/extension_point.h"

namespace cudaq {
inline std::optional<std::unordered_map<std::string, std::any>::const_iterator>
findIter(const std::vector<std::string> &possibleNames,
         const std::unordered_map<std::string, std::any> &m) {
  for (auto &name : possibleNames) {
    auto iter = m.find(name);
    if (iter != m.end())
      return iter;
  }
  return std::nullopt;
}

inline std::size_t getIntLike(const std::any &any) {
  try {
    return std::any_cast<std::size_t>(any);
  } catch (...) {
    // If this throws then we'll just have an error
    return std::any_cast<int>(any);
  }
}

class operator_pool : public extension_point<operator_pool> {
public:
  operator_pool() = default;
  virtual ~operator_pool() {}
  virtual std::vector<spin_op>
  generate(const std::unordered_map<std::string, std::any> &config) const = 0;
};

CUDAQ_DEFINE_EXTENSION_IMPL(operator_pool)

#define CUDAQ_REGISTER_OPERATOR_POOL(TYPE)                                     \
  static inline const std::string class_identifier = #TYPE;                    \
  static std::unique_ptr<operator_pool> create() {                             \
    return std::make_unique<TYPE>();                                           \
  }

} // namespace cudaq
