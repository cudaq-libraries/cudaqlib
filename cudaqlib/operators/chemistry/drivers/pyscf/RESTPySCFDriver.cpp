/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nlohmann/json.hpp"

#include "cudaqlib/operators/chemistry/MoleculePackageDriver.h"
#include "cudaqlib/operators/fermion/fermion_to_spin.h"
#include "cudaqlib/utils/library_utils.h"

#include "common/RestClient.h"

#include <filesystem>
#include <fmt/core.h>
#include <fstream>

namespace cudaq::operators {

class RESTPySCFDriver
    : public details::MoleculePackageDriver_impl<RESTPySCFDriver> {

public:
  bool is_available() const override {
    RestClient client;
    std::map<std::string, std::string> headers;
    try {
      auto res = client.get("localhost:8000/", "status", headers);
      if (res.contains("status") &&
          res["status"].get<std::string>() == "available")
        return true;
    } catch (std::exception &e) {
      return false;
    }
    return true;
  }

  /// @brief Create the molecular hamiltonian
  operators::molecular_hamiltonian
  createMolecule(const molecular_geometry &geometry, const std::string &basis,
                 int spin, int charge, molecule_options options) override {
    std::string xyzFileStr = "";
    // Convert the geometry to an XYZ string
    for (auto &atom : geometry)
      xyzFileStr +=
          fmt::format("{} {:f} {:f} {:f}; ", atom.name, atom.coordinates[0],
                      atom.coordinates[1], atom.coordinates[2]);

    RestClient client;
    nlohmann::json payload = {{"xyz", xyzFileStr},
                              {"basis", basis},
                              {"spin", spin},
                              {"charge", charge},
                              {"type", "gas_phase"},
                              {"symmetry", false},
                              {"cycles", options.cycles},
                              {"initguess", options.initguess},
                              {"UR", options.UR},
                              {"MP2", options.MP2},
                              {"natorb", options.natorb},
                              {"casci", options.casci},
                              {"ccsd", options.ccsd},
                              {"casscf", options.casscf},
                              {"integrals_natorb", options.integrals_natorb},
                              {"integrals_casscf", options.integrals_casscf},
                              {"verbose", options.verbose}};
    if (options.nele_cas.has_value())
      payload["nele_cas"] = options.nele_cas.value();
    if (options.norb_cas.has_value())
      payload["norb_cas"] = options.norb_cas.value();
    if (options.potfile.has_value())
      payload["potfile"] = options.potfile.value();

    std::map<std::string, std::string> headers{
        {"Content-Type", "application/json"}};
    auto metadata = client.post("localhost:8000/", "create_molecule", payload,
                                headers, true);

    // Get the energy, num orbitals, and num qubits
    std::unordered_map<std::string, double> energies;
    for (auto &[energyName, E] : metadata["energies"].items())
      energies.insert({energyName, E});

    double energy = 0.0;
    if (energies.contains("nuclear_energy"))
      energy = energies["nuclear_energy"];
    else if (energies.contains("core_energy"))
      energy = energies["core_energy"];

    auto numOrb = metadata["num_orbitals"].get<std::size_t>();
    auto numQubits = 2 * numOrb;
    auto num_electrons = metadata["num_electrons"].get<std::size_t>();

    // Get the operators
    fermion_op fermionOp(numQubits, energy);
    std::unordered_map<std::string, fermion_op> operators;
    auto hpqElements = metadata["hpq"]["data"];
    auto hpqrsElements = metadata["hpqrs"]["data"];
    std::vector<std::complex<double>> hpqValues, hpqrsValues;
    for (auto &element : hpqElements)
      hpqValues.push_back({element[0].get<double>(), element[1].get<double>()});
    for (auto &element : hpqrsElements)
      hpqrsValues.push_back(
          {element[0].get<double>(), element[1].get<double>()});
    fermionOp.hpq.set_data(hpqValues);
    fermionOp.hpqrs.set_data(hpqrsValues);

    // Transform to a spin operator
    auto transform = fermion_to_spin::get(options.fermion_to_string);
    auto spinHamiltonian = transform->generate(fermionOp);

    // Return the molecular hamiltonian
    return operators::molecular_hamiltonian{
        spinHamiltonian, std::move(fermionOp), num_electrons, numOrb, energies};
  }

  CUDAQ_REGISTER_MOLECULEPACKAGEDRIVER(RESTPySCFDriver)
};

} // namespace cudaq::operators