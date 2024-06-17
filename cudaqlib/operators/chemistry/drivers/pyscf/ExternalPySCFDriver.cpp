/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "json.hpp"

#include "cudaqlib/operators/chemistry/MoleculePackageDriver.h"
#include "cudaqlib/operators/fermion/fermion_to_spin.h"
#include "cudaqlib/utils/library_utils.h"

#include <filesystem>
#include <fmt/core.h>
#include <fstream>

namespace cudaq::operators {

class external_pyscf
    : public details::MoleculePackageDriver_impl<external_pyscf> {
protected:
  /// @brief Extract the coefficient data from the given binary file/
  std::vector<std::complex<double>> readInData(const std::string &file) {
    std::ifstream input(file, std::ios::binary);
    if (input.fail())
      throw std::runtime_error(file + " does not exist.");

    input.seekg(0, std::ios_base::end);
    std::size_t size = input.tellg();
    input.seekg(0, std::ios_base::beg);
    std::vector<std::complex<double>> input_vec(size /
                                                sizeof(std::complex<double>));
    input.read((char *)&input_vec[0], size);
    return input_vec;
  }

public:
  /// @brief Create the molecular hamiltonian
  operators::molecular_hamiltonian
  createMolecule(const molecular_geometry &geometry, const std::string &basis,
                 int spin, int charge, molecule_options options) override {
    std::string toolPath, xyzFileStr = "", outFileName = "tmpFileToBeDeleted";
    std::string oneBodyFile = outFileName + "_one_body.dat";
    std::string twoBodyFile = outFileName + "_two_body.dat";
    std::string metadataFile = outFileName + "_metadata.json";
    std::filesystem::path libPath{cudaq::__internal__::getCUDAQLibraryPath()};
    auto cudaqLibPath = libPath.parent_path();
    auto cudaqPySCFTool =
        cudaqLibPath.parent_path() / "bin" / "cudaq-pyscf.py";

    // Convert the geometry to an XYZ string
    for (auto &atom : geometry)
      xyzFileStr +=
          fmt::format("{} {:f} {:f} {:f}; ", atom.name, atom.coordinates[0],
                      atom.coordinates[1], atom.coordinates[2]);
    xyzFileStr = "\"" + xyzFileStr + "\"";

    std::string argString = fmt::format(
        "{} --type {} --xyz {} --charge {} --spin {} --basis {} --out-file-name {} "
        "{} --memory {} {} --cycles {} --initguess {} {} {} {} {} {} {} {} {} "
        "{} {} {}",
        cudaqPySCFTool.string(), options.type, xyzFileStr, charge, spin, basis, outFileName,
        options.verbose ? "--verbose" : "", options.memory,
        options.symmetry ? "--symmetry" : "", options.cycles, options.initguess,
        options.UR ? "--UR" : "",
        options.nele_cas.has_value()
            ? "--nele_cas " + std::to_string(options.nele_cas.value())
            : "",
        options.norb_cas.has_value()
            ? "--norb_cas " + std::to_string(options.norb_cas.value())
            : "",
        options.MP2 ? "--MP2" : "", options.natorb ? "--natorb" : "",
        options.casci ? "--casci" : "", options.ccsd ? "--ccsd" : "",
        options.casscf ? "--casscf" : "",
        options.integrals_natorb ? "--integrals_natorb" : "",
        options.integrals_casscf ? "--integrals_casscf" : "",
        options.potfile.has_value() ? "--potfile " + options.potfile.value()
                                    : "");

    // Run the external pyscf script
    auto ret = std::system(argString.c_str());
    if (ret != 0)
      throw std::runtime_error("Failed to generate molecular data with pyscf");

    // Import all the data we need from the execution.
    std::ifstream f(metadataFile);
    auto metadata = nlohmann::json::parse(f);
    metadata.dump();
    auto oneVec = readInData(oneBodyFile);
    auto twoVec = readInData(twoBodyFile);
    std::remove(metadataFile.c_str());
    std::remove(oneBodyFile.c_str());
    std::remove(twoBodyFile.c_str());

    // Get the energy, num orbitals, and num qubits
    auto energy = metadata["nuclear_energy"].get<double>();
    auto numOrb = metadata["num_orbitals"].get<std::size_t>();
    auto numQubits = 2 * numOrb;

    // Create the fermion operator
    fermion_op fermionOp(numQubits, energy);
    fermionOp.hpq.set_data(oneVec);
    fermionOp.hpqrs.set_data(twoVec);

    // Transform to a spin operator
    auto transform = fermion_to_spin::get(options.fermion_to_string);
    auto spinHamiltonian = transform->generate(fermionOp);

    // Return the molecular hamiltonian
    return operators::molecular_hamiltonian{
        spinHamiltonian,
        std::move(fermionOp),
        metadata["num_electrons"].get<std::size_t>(),
        numOrb,
        energy,
        metadata["hf_energy"].get<double>()};
  }

  CUDAQX_REGISTER_MOLECULEPACKAGEDRIVER(external_pyscf)
};

} // namespace cudaq::operators