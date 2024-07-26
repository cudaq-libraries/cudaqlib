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

  operators.def("jordan_wigner", [](fermion_op &op) {
    return fermion_compiler::get("jordan_wigner")->generate(op);
  });

  py::class_<fermion_compiler>(operators, "fermion_compiler")
      .def_static(
          "get",
          [](const std::string &name) { return fermion_compiler::get(name); })
      .def("generate", &fermion_compiler::generate, "");

  py::class_<one_body_integrals>(operators, "OneBodyIntegrals",
                                 py::buffer_protocol())
      .def_buffer([](one_body_integrals &m) -> py::buffer_info {
        return py::buffer_info(
            m.raw_data(),                 /* Pointer to buffer */
            sizeof(std::complex<double>), /* Size of one scalar */
            py::format_descriptor<std::complex<double>>::format(), /* Python
                                                       struct-style format
                                                       descriptor */
            2,       /* Number of dimensions */
            m.shape, /* Buffer dimensions */
            {sizeof(std::complex<double>) *
                 m.shape[1], /* Strides (in bytes) for each index */
             sizeof(std::complex<double>)});
      });

  py::class_<two_body_integrals>(operators, "TwoBodyIntegrals",
                                 py::buffer_protocol())
      .def_buffer([](two_body_integrals &m) -> py::buffer_info {
        auto calculateStrides = [](std::vector<std::size_t> &shape_) {
          std::vector<size_t> strides(4);
          strides[3] = sizeof(std::complex<double>);
          strides[2] = strides[3] * shape_[3];
          strides[1] = strides[2] * shape_[2];
          strides[0] = strides[1] * shape_[1];
          return strides;
        };
        return py::buffer_info(
            m.raw_data(),                 /* Pointer to buffer */
            sizeof(std::complex<double>), /* Size of one scalar */
            py::format_descriptor<std::complex<double>>::format(), /* Python
                                                       struct-style format
                                                       descriptor */
            4,       /* Number of dimensions */
            m.shape, /* Buffer dimensions */
            calculateStrides(m.shape));
      });

  py::class_<fermion_op>(operators, "FermionOperator", "")
      .def_readonly("hpq", &fermion_op::hpq, "")
      .def_readonly("hpqrs", &fermion_op::hpqrs, "");

  py::class_<molecular_hamiltonian>(operators, "MolecularHamiltonian")
      .def_readonly("energies", &molecular_hamiltonian::energies)
      .def_readonly("hamiltonian", &molecular_hamiltonian::hamiltonian)
      .def_readonly("n_electrons", &molecular_hamiltonian::n_electrons)
      .def_readonly("n_orbitals", &molecular_hamiltonian::n_orbitals);

  operators.def(
      "one_particle_op",
      [](std::size_t numQ, std::size_t p, std::size_t q) {
        return operators::one_particle_op(numQ, p, q);
      },
      "");

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