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
        return create_molecule(molGeom, basis, spin, charge);
      },
      "");
}

} // namespace cudaq::operators