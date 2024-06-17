/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaqlib/utils/extension_point.h"
#include "molecule.h"

namespace cudaq::operators {

/// @brief MoleculePackageDriver provides an extensible interface for
/// generating molecular Hamiltonians and associated metadata.
class MoleculePackageDriver : public extension_point<MoleculePackageDriver> {
public:
  /// @brief Return a `molecular_hamiltonian` described by the given
  /// geometry, basis set, spin, and charge. Optionally
  /// restrict the active space.
  virtual molecular_hamiltonian
  createMolecule(const molecular_geometry &geometry, const std::string &basis,
                 int spin, int charge,
                 molecule_options options = molecule_options()) = 0;

  /// Virtual destructor needed when deleting an instance of a derived class
  /// via a pointer to the base class.
  virtual ~MoleculePackageDriver(){};
};

CUDAQ_DEFINE_EXTENSION_IMPL(MoleculePackageDriver)

#define CUDAQ_REGISTER_MOLECULEPACKAGEDRIVER(TYPE)                            \
  static inline const std::string class_identifier = #TYPE;                    \
  static std::unique_ptr<MoleculePackageDriver> create() {                     \
    return std::make_unique<TYPE>();                                           \
  }
} // namespace cudaq::operators
