/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace cudaq {

/// @brief Return the value of given type corresponding to the provided
/// key string from the provided options `kwargs` `dict`. Return the `orVal`
/// if the key is not in the `dict`.
template <typename T>
T getValueOr(py::kwargs &options, const std::string &key, const T &orVal) {
  if (options.contains(key))
    for (auto item : options)
      if (item.first.cast<std::string>() == key)
        return item.second.cast<T>();

  return orVal;
}

} // namespace cudaq