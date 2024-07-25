/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <optional>

#include "cudaqlib/utils/extension_point.h"

namespace cudaq::optim {

/// Typedef modeling the result of an optimization strategy,
/// a double representing the optimal value and the corresponding
/// optimal parameters.
using optimization_result = std::tuple<double, std::vector<double>>;

/// An optimizable_function wraps a user-provided objective function
/// to be optimized.
class optimizable_function {
private:
  // Useful typedefs
  using NoGradientSignature =
      std::function<double(const std::vector<double> &)>;
  using GradientSignature =
      std::function<double(const std::vector<double> &, std::vector<double> &)>;

  // The function we are optimizing
  GradientSignature _opt_func;
  bool _providesGradients = true;

public:
  template <typename Callable>
  optimizable_function(const Callable &callable) {
    static_assert(
        std::is_invocable_v<Callable, std::vector<double>> ||
            std::is_invocable_v<Callable, std::vector<double>,
                                std::vector<double> &>,
        "Invalid optimization function. Must have signature double(const "
        "std::vector<double>&) or double(const std::vector<double>&, "
        "std::vector<double>&) for gradient-free or gradient-based "
        "optimizations, respectively.");

    if constexpr (std::is_invocable_v<Callable, std::vector<double>>) {
      _opt_func = [c = std::move(callable)](const std::vector<double> &x,
                                            std::vector<double> &) {
        return c(x);
      };
      _providesGradients = false;
    } else {
      _opt_func = std::move(callable);
    }
  }

  bool providesGradients() const { return _providesGradients; }
  double operator()(const std::vector<double> &x,
                    std::vector<double> &dx) const {
    return _opt_func(x, dx);
  }
};

struct optimizer_options {
  /// @brief Set the number of shots for
  /// objective functions requiring quantum evaluation.
  int shots = -1;

  /// @brief The maximum number of optimization iterations.
  std::optional<std::size_t> max_iterations;

  /// @brief The initial parameters at which to begin optimization.
  std::optional<std::vector<double>> initial_parameters;

  /// @brief The function tolerance.
  std::optional<double> function_tolerance;

  /// @brief Turn on verbose mode
  bool verbose = false;
};

///
/// The cudaq::optimizer provides a high-level interface for general
/// optimization of user-specified objective functions. This is meant
/// to serve an interface for clients working with concrete
/// subtypes providing specific optimization algorithms possibly delegating
/// to third party libraries. This interface provides an optimize(...) method
/// that takes the number of objective function input parameters
/// (the dimension), and a user-specified objective function that takes the
/// function input parameters as a immutable (const) vector<double> reference
/// and a mutable vector<double> reference modeling the current iteration
/// gradient vector (df / dx_i, for all i parameters). This function
/// must return a scalar double, the value of this function at the
/// current input parameters. The optimizer also
/// exposes a method for querying whether the current optimization strategy
/// requires gradients or not. Parameterizing optimization strategies
/// is left as a task for sub-types (things like initial parameters, max
/// function evaluations, etc.).
class optimizer : public extension_point<optimizer> {
public:
  virtual ~optimizer() = default;

  /// Returns true if this optimization strategy requires
  /// gradients to achieve its optimization goals.
  virtual bool requiresGradients() = 0;

  /// Run the optimization strategy defined by concrete sub-type
  /// implementations. Takes the number of variational parameters,
  /// and a custom objective function that takes the
  /// function input parameters as a immutable (`const`) `vector<double>`
  /// reference and a mutable `vector<double>` reference modeling the current
  /// iteration gradient vector (`df / dx_i`, for all `i` parameters). This
  /// function must return a scalar double, the value of this function at the
  /// current input parameters.
  virtual optimization_result
  optimize(std::size_t dim, const optimizable_function &opt_function) {
    return optimize(dim, optimizer_options(), opt_function);
  }

  /// Run the optimization strategy defined by concrete sub-type
  /// implementations. Takes the number of variational parameters,
  /// and a custom objective function that takes the
  /// function input parameters as a immutable (`const`) `vector<double>`
  /// reference and a mutable `vector<double>` reference modeling the current
  /// iteration gradient vector (`df / dx_i`, for all `i` parameters). This
  /// function must return a scalar double, the value of this function at the
  /// current input parameters. Optionally provide optimizer options.
  virtual optimization_result
  optimize(std::size_t dim, const optimizer_options &options,
           const optimizable_function &opt_function) = 0;

  /// @brief Return the optimization history, e.g.
  /// the value and parameters found for each iteration of
  /// the optimization.
  std::vector<optimization_result> history;
};

CUDAQ_DEFINE_EXTENSION_IMPL(optimizer)

#define CUDAQ_REGISTER_OPTIMIZER(TYPE)                                         \
  static inline const std::string class_identifier = #TYPE;                    \
  static std::unique_ptr<optimizer> create() {                                 \
    return std::make_unique<TYPE>();                                           \
  }

} // namespace cudaq::optim
