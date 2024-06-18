/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <limits>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaqlib/operators.h"
#include "../utils/type_casters.h"
#include "../utils/utils.h"

namespace py = pybind11;

namespace cudaq::operators {

void bindOperators(py::module &mod) {

  auto operators = mod.def_submodule("operators");
  py::class_<molecular_hamiltonian>(operators, "MolecularHamiltonian")
      .def_readonly("hamiltonian", &molecular_hamiltonian::hamiltonian)
      .def_readonly("n_electrons", &molecular_hamiltonian::n_electrons)
      .def_readonly("n_orbitals", &molecular_hamiltonian::n_orbitals)
      .def_readonly("nuclear_repulsion",
                    &molecular_hamiltonian::nuclear_repulsion)
      .def_readonly("hf_energy", &molecular_hamiltonian::hf_energy)
      .def_readonly("fci_energy", &molecular_hamiltonian::fci_energy);

  operators.def(
      "create_molecule",
      [](py::list geometry, const std::string basis, int spin, int charge,
         py::kwargs options) {
        std::vector<atom> atoms;
        for (auto el : geometry) {
          if (!py::isinstance<py::tuple>(el))
            throw std::runtime_error(
                "geometry must be a list of tuples ('NAME', (X, Y, Z))");
          auto casted = el.cast<py::tuple>();
          if (!py::isinstance<py::tuple>(casted[1]))
            throw std::runtime_error(
                "geometry must be a list of tuples ('NAME', (X, Y, Z))");

          auto name = casted[0].cast<std::string>();
          auto coords = casted[1].cast<py::tuple>();
          atoms.push_back(
              atom{name,
                   {coords[0].cast<double>(), coords[1].cast<double>(),
                    coords[2].cast<double>()}});
        }
        molecular_geometry molGeom(atoms);

        molecule_options inOptions;
        inOptions.type = getValueOr<std::string>(options, "type", "gas_phase");
        std::optional<std::size_t> nele_cas =
            getValueOr<std::size_t>(options, "nele_cas", -1);
        inOptions.nele_cas = nele_cas == -1 ? std::nullopt : nele_cas;
        std::optional<std::size_t> norb_cas =
            getValueOr<std::size_t>(options, "norb_cas", -1);
        inOptions.norb_cas = norb_cas == -1 ? std::nullopt : norb_cas;
        inOptions.symmetry = getValueOr<bool>(options, "symmetry", false);
        inOptions.memory = getValueOr<double>(options, "memory", 4000.);
        inOptions.cycles = getValueOr<std::size_t>(options, "cycles", 100);
        inOptions.initguess =
            getValueOr<std::string>(options, "initguess", "minao");
        inOptions.UR = getValueOr<bool>(options, "UR", false);
        inOptions.MP2 = getValueOr<bool>(options, "MP2", false);
        inOptions.natorb = getValueOr<bool>(options, "natorb", false);
        inOptions.casci = getValueOr<bool>(options, "casci", false);
        inOptions.ccsd = getValueOr<bool>(options, "ccsd", false);
        inOptions.casscf = getValueOr<bool>(options, "casscf", false);
        inOptions.integrals_natorb =
            getValueOr<bool>(options, "integrals_natorb", false);
        inOptions.integrals_casscf =
            getValueOr<bool>(options, "integrals_casscf", false);
        inOptions.verbose = getValueOr<bool>(options, "verbose", false);

        if (inOptions.verbose)
          inOptions.dump();
        return create_molecule(molGeom, basis, spin, charge, inOptions);
      },
      "");
}

} // namespace cudaq::operators