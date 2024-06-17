/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "lbfgs.h"
#include "LBFGSObjective.h"

namespace cudaq::optim {
optimization_result lbfgs::optimize(std::size_t dim,
                                    const optimizer_options &options,
                                    const optimizable_function &opt_function) {
  history.clear();
  LBFGSObjective f(
      opt_function,
      options.initial_parameters.value_or(std::vector<double>(dim)),
      options.function_tolerance.value_or(1e-12),
      options.max_iterations.value_or(std::numeric_limits<std::size_t>::max()),
      history, options.verbose);
  return f.run(dim);
}

} // namespace cudaq::optim