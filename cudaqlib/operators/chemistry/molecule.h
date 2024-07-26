/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/spin_op.h"
#include "cudaqlib/operators/fermion/fermion_op.h"

#include <optional>

namespace cudaq::operators {

/// @brief An atom represents an atom from the
/// periodic table, it has a name and a coordinate in 3D space.
struct atom {
  const std::string name;
  const double coordinates[3];
};

/// @brief The molecular_geometry encapsulates a vector
/// of atoms, each describing their name and location in
/// 3D space.
class molecular_geometry {
private:
  std::vector<atom> atoms;

public:
  molecular_geometry(std::initializer_list<atom> &&args)
      : atoms(args.begin(), args.end()) {}
  molecular_geometry(const std::vector<atom> &args) : atoms(args) {}
  std::size_t size() const { return atoms.size(); }
  auto begin() { return atoms.begin(); }
  auto end() { return atoms.end(); }
  auto begin() const { return atoms.cbegin(); };
  auto end() const { return atoms.cend(); }
  std::string name() const;
  static molecular_geometry from_xyz(const std::string &xyzFile);
};

/// @brief The `molecular_hamiltonian` type holds all the pertinent
/// data for a molecule created by CUDA Quantum from its geometry and
/// other metadata.
struct molecular_hamiltonian {
  spin_op hamiltonian;
  operators::fermion_op fermionOperator;
  std::size_t n_electrons;
  std::size_t n_orbitals;
  std::unordered_map<std::string, double> energies;
};

struct molecule_options {
  std::string driver = "RESTPySCFDriver";
  std::string fermion_to_string = "jordan_wigner";
  std::string type = "gas_phase";
  bool symmetry = false;
  double memory = 4000.;
  std::size_t cycles = 100;
  std::string initguess = "minao";
  bool UR = false;
  std::optional<std::size_t> nele_cas = std::nullopt;
  std::optional<std::size_t> norb_cas = std::nullopt;
  bool MP2 = false;
  bool natorb = false;
  bool casci = false;
  bool ccsd = false;
  bool casscf = false;
  bool integrals_natorb = false;
  bool integrals_casscf = false;
  std::optional<std::string> potfile = std::nullopt;
  bool verbose = false;
  void dump();
};

/// @brief Given a molecular structure and other metadata,
/// construct the Hamiltonian for the molecule as a `cudaq::spin_op`
molecular_hamiltonian
create_molecule(const molecular_geometry &geometry, const std::string &basis,
                int spin, int charge,
                molecule_options options = molecule_options());

spin_op one_particle_op(std::size_t numQubits, std::size_t p, std::size_t q,
                        const std::string fermionCompiler = "jordan_wigner");

} // namespace cudaq::operators
