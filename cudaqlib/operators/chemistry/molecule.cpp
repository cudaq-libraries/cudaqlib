/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "molecule.h"
#include "MoleculePackageDriver.h"

#include <fstream>
#include <iostream>

namespace cudaq::operators {

std::string molecular_geometry::name() const {
  std::string ret = "";
  for (auto &a : atoms)
    ret += a.name;

  return ret;
}

molecular_geometry molecular_geometry::from_xyz(const std::string &xyzFile) {

  std::ifstream stream(xyzFile);
  std::string contents((std::istreambuf_iterator<char>(stream)),
                       std::istreambuf_iterator<char>());

  if (contents.empty())
    throw std::runtime_error("could not extract file contents for " + xyzFile);
  auto lines = cudaq::split(contents, '\n');
  char *endptr;
  int result = std::strtol(lines[0].c_str(), &endptr, 10);

  std::vector<atom> atoms;
  for (std::size_t i = 0; auto &line : lines) {
    if (i++ == 0)
      continue;

    cudaq::trim(line);
    if (line.empty())
      continue;

    bool seenFirstSpace = false;
    std::vector<std::string> components(/*atom+3coord*/ 4);
    for (std::size_t k = 0, componentCounter = 0; k < line.length(); k++) {
      if (line[k] == ' ') {
        if (!seenFirstSpace) {
          seenFirstSpace = true;
          components[componentCounter] += " ";
          componentCounter++;
        }
        continue;
      }

      seenFirstSpace = false;
      components[componentCounter] += line[k];
    }

    std::vector<double> coords;
    for (std::size_t ii = 1; ii < 4; ii++)
      coords.push_back(std::stod(components[ii]));
    atoms.push_back(atom{components[0], {coords[0], coords[1], coords[2]}});
  }

  return molecular_geometry(atoms);
}

molecular_hamiltonian create_molecule(const molecular_geometry &geometry,
                                      const std::string &basis, int spin,
                                      int charge, molecule_options options) {
  if (!MoleculePackageDriver::is_registered(options.driver))
    throw std::runtime_error("invalid molecule package driver (" +
                             options.driver + ")");
  return MoleculePackageDriver::get(options.driver)
      ->createMolecule(geometry, basis, spin, charge, options);
}

void molecule_options::dump() {
  std::cout << "\tmolecule_options dump() [\n";
  std::cout << "\tfermion_to_string: " << fermion_to_string << "\n";
  std::cout << "\ttype: " << type << "\n";
  std::cout << "\tsymmetry: " << symmetry << "\n";
  std::cout << "\tcycles: " << cycles << "\n";
  std::cout << "\tinitguess: " << initguess << "\n";
  std::cout << "\tnele_cas: " << (nele_cas.has_value() ? nele_cas.value()
                                                      : -1) << "\n";
  std::cout << "\tnorb_cas: " << (norb_cas.has_value() ? norb_cas.value()
                                                      : -1) << "\n";
  std::cout << "\tUR: " << std::boolalpha << UR << "\n";
  std::cout << "\tMP2: " << std::boolalpha << MP2 << "\n";
  std::cout << "\tnatorb: " << std::boolalpha << natorb << "\n";
  std::cout << "\tcasci: " << std::boolalpha << casci << "\n";
  std::cout << "\tccsd: " << std::boolalpha << ccsd << "\n";
  std::cout << "\tcasscf: " << std::boolalpha << casscf << "\n";
  std::cout << "\tintegrals_natorb: " << std::boolalpha << integrals_natorb
            << "\n";
  std::cout << "\tintegrals_casscf: " << std::boolalpha << integrals_casscf
            << "\n";
  std::cout << "\tintegrals_natorb: " << std::boolalpha << integrals_natorb
            << "\n]\n";
}
} // namespace cudaq::operators
