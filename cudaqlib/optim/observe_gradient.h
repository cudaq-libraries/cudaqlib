#pragma once

#include "cudaq/algorithms/observe.h"
#include "optimizer.h"

namespace cudaq::optim {

/// Parameterized kernels for observe optimization must take on this
/// signature - take a vector of double and return void.
using ParameterizedKernel = std::function<void(std::vector<double>)>;

/// @brief Observe executions can be used for function
/// evaluation or gradient evaluation. This enumeration
/// allows us to distinguish these execution types.
enum class observe_execution_type { function, gradient };

/// @brief Storage type for a single observe iteration. Keeps
/// track of the parameters evaluated at, the result of the
/// observation (shots data and expectations), and the type of
/// the execution.
struct observe_iteration {
  std::vector<double> parameters;
  observe_result result;
  observe_execution_type type;
};

/// @brief The observe_gradient provides an extension point
/// for developers to inject custom gradient strategies to be
/// used in global optimization of expectation values in
/// typical quantum variational tasks.
class observe_gradient : public extension_point<observe_gradient> {
protected:
  /// The spin operator used in computing expectation values
  /// via `cudaq::observe`
  spin_op op;

  /// The parameterized CUDA Quantum kernel function.
  ParameterizedKernel quantumFunction;

  /// The current batch expectation value to be computed
  std::size_t batchIdx = 0;

  /// The total number of expectation values required
  std::size_t numRequiredExpectations = 0;

  /// @brief The number of shots for expectation value computation
  int shots = -1;

  /// @brief Compute the expectation value at the given parameters.
  double expectation(std::vector<double> &x) {
    auto &platform = cudaq::get_platform();
    std::string kernelName =
        "auto_gradient_kernel_calc_" + std::to_string(batchIdx);
    auto result = cudaq::details::runObservation(
        [&]() { quantumFunction(x); }, const_cast<spin_op &>(op), platform,
        shots, kernelName, 0, nullptr, batchIdx, numRequiredExpectations);
    data.emplace_back(x, result.value(), observe_execution_type::gradient);
    batchIdx++;
    return result.value().expectation();
  }

  /// @brief Compute the gradient at the given multi-dimensional point.
  /// The gradient vector is provided as a non-const reference, subtypes
  /// therefore should update the vector in place.
  virtual void calculateGradient(const std::vector<double> &x,
                                 std::vector<double> &dx,
                                 double expectationAtX) = 0;

  /// @brief Return the number of expectation computations required to
  /// compute the gradient, e.g. 2 for a single parameter parameter-shift rule.
  virtual std::size_t
  getRequiredNumExpectationComputations(const std::vector<double> &x) = 0;

public:
  observe_gradient() = default;

  /// The constructor
  observe_gradient(const ParameterizedKernel &functor, const spin_op &op)
      : op(op), quantumFunction(functor) {}

  /// @brief Storage for all data produced during gradient computation.
  std::vector<observe_iteration> data;

  void set_spin_op(const spin_op in_op) { op = in_op; }
  void set_parameterized_kernel(const ParameterizedKernel kernel) {
    quantumFunction = kernel;
  }

  /// @brief Compute the gradient at the given multi-dimensional point.
  /// The gradient vector is provided as a non-const reference, subtypes
  /// therefore should update the vector in place. This delegates to specific
  /// subtype implementations. It tracks the number of expectations that
  /// need to be computed, and executes them as a batch (e.g. allocates the
  /// state once, zeros the state between each iteration instead of
  /// deallocating).
  void compute(const std::vector<double> &x, std::vector<double> &dx,
               double expectationAtX, int inShots = -1) {
    if (!quantumFunction)
      throw std::runtime_error("[observe_gradient] kernel function not set.");

    shots = inShots;
    numRequiredExpectations = getRequiredNumExpectationComputations(x);
    calculateGradient(x, dx, expectationAtX);
    // reset
    numRequiredExpectations = 0;
    batchIdx = 0;
  }

  virtual ~observe_gradient() {}
};

CUDAQ_DEFINE_EXTENSION_IMPL(observe_gradient)

#define CUDAQ_REGISTER_GRADIENT(TYPE)                                          \
  using details::observe_gradient_impl<TYPE>::observe_gradient_impl;           \
  static inline const std::string class_identifier = #TYPE;                    \
  static std::unique_ptr<observe_gradient> create() {                          \
    return std::make_unique<TYPE>();                                           \
  }

} // namespace cudaq::optim