/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include "cudaq/utils/cudaq_utils.h"

#include <cmath>
#include <tuple>
#include <vector>

namespace cudaq {
using excitation_list = std::vector<std::vector<std::size_t>>;

std::tuple<excitation_list, excitation_list, excitation_list, excitation_list,
           excitation_list>
get_uccsd_excitations(std::size_t numElectrons, std::size_t spin,
                      std::size_t numQubits) {
  if (numQubits % 2 != 0)
    throw std::runtime_error("The total number of qubits should be even.");

  auto numSpatialOrbs = numQubits / 2;
  std::vector<std::size_t> occupiedAlpha, virtualAlpha, occupiedBeta,
      virtualBeta;
  if (spin > 0) {
    auto n_occupied_beta =
        static_cast<std::size_t>(std::floor((float)(numElectrons - spin) / 2));
    auto n_occupied_alpha = numElectrons - n_occupied_beta;
    auto n_virtual_alpha = numSpatialOrbs - n_occupied_alpha;
    auto n_virtual_beta = numSpatialOrbs - n_occupied_beta;

    for (auto i : cudaq::range(n_occupied_alpha))
      occupiedAlpha.push_back(i * 2);

    for (auto i : cudaq::range(n_virtual_alpha))
      virtualAlpha.push_back(i * 2 + numElectrons + 1);

    for (auto i : cudaq::range(n_occupied_beta))
      occupiedBeta.push_back(i * 2 + 1);

    for (auto i : cudaq::range(n_virtual_beta))
      virtualBeta.push_back(i * 2 + numElectrons - 1);
  } else if (numElectrons % 2 == 0 && spin == 0) {
    auto numOccupied =
        static_cast<std::size_t>(std::floor((float)numElectrons / 2));
    auto numVirtual = numSpatialOrbs - numOccupied;

    for (auto i : cudaq::range(numOccupied))
      occupiedAlpha.push_back(i * 2);

    for (auto i : cudaq::range(numVirtual))
      virtualAlpha.push_back(i * 2 + numElectrons);

    for (auto i : cudaq::range(numOccupied))
      occupiedBeta.push_back(i * 2 + 1);

    for (auto i : cudaq::range(numVirtual))
      virtualBeta.push_back(i * 2 + numElectrons + 1);

  } else
    throw std::runtime_error("Incorrect spin multiplicity. Number of electrons "
                             "is odd but spin is 0.");

  excitation_list singlesAlpha, singlesBeta, doublesMixed, doublesAlpha,
      doublesBeta;

  for (auto p : occupiedAlpha)
    for (auto q : virtualAlpha)
      singlesAlpha.push_back({p, q});

  for (auto p : occupiedBeta)
    for (auto q : virtualBeta)
      singlesBeta.push_back({p, q});

  for (auto p : occupiedAlpha)
    for (auto q : occupiedBeta)
      for (auto r : virtualBeta)
        for (auto s : virtualAlpha)
          doublesMixed.push_back({p, q, r, s});

  auto numOccAlpha = occupiedAlpha.size();
  auto numOccBeta = occupiedBeta.size();
  auto numVirtAlpha = virtualAlpha.size();
  auto numVirtBeta = virtualBeta.size();

  for (auto p : cudaq::range(numOccAlpha - 1))
    for (std::size_t q = p + 1; q < numOccAlpha; q++)
      for (auto r : cudaq::range(numVirtAlpha - 1))
        for (std::size_t s = r + 1; s < numVirtAlpha; s++)
          doublesAlpha.push_back({occupiedAlpha[p], occupiedAlpha[q],
                                  virtualAlpha[r], virtualAlpha[s]});

  for (auto p : cudaq::range(numOccBeta - 1))
    for (std::size_t q = p + 1; q < numOccBeta; q++)
      for (auto r : cudaq::range(numVirtBeta - 1))
        for (std::size_t s = r + 1; s < numVirtBeta; s++)
          doublesBeta.push_back({occupiedBeta[p], occupiedBeta[q],
                                 virtualBeta[r], virtualBeta[s]});

  return std::make_tuple(singlesAlpha, singlesBeta, doublesMixed, doublesAlpha,
                         doublesBeta);
}

auto uccsd_num_parameters(std::size_t numElectrons, std::size_t numQubits, std::size_t spin = 0) {
  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      get_uccsd_excitations(numElectrons, spin, numQubits);
  return singlesAlpha.size() + singlesBeta.size() + doublesMixed.size() +
         doublesAlpha.size() + doublesBeta.size();
}

__qpu__ void singleExcitation(cudaq::qview<> qubits, std::size_t pOcc,
                              std::size_t qVirt, double theta) {
  // Y_p X_q
  rx(M_PI_2, qubits[pOcc]);
  h(qubits[qVirt]);

  for (std::size_t i = pOcc; i < qVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.5 * theta, qubits[qVirt]);

  for (std::size_t i = qVirt; i > pOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[qVirt]);
  rx(-M_PI_2, qubits[pOcc]);

  // -X_p Y_q
  h(qubits[pOcc]);
  rx(M_PI_2, qubits[qVirt]);

  for (std::size_t i = pOcc; i < qVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.5 * theta, qubits[qVirt]);

  for (std::size_t i = qVirt; i > pOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[qVirt]);
  h(qubits[pOcc]);
}

template <typename Kernel>
void singleExcitation(Kernel &kernel, QuakeValue &qubits, std::size_t pOcc,
                      std::size_t qVirt, QuakeValue &theta) {
  // Y_p X_q
  kernel.rx(M_PI_2, qubits[pOcc]);
  kernel.h(qubits[qVirt]);

  for (std::size_t i = pOcc; i < qVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.5 * theta, qubits[qVirt]);

  for (std::size_t i = qVirt; i > pOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.h(qubits[qVirt]);
  kernel.rx(-M_PI_2, qubits[pOcc]);

  // -X_p Y_q
  kernel.h(qubits[pOcc]);
  kernel.rx(M_PI_2, qubits[qVirt]);

  for (std::size_t i = pOcc; i < qVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.5 * theta, qubits[qVirt]);

  for (std::size_t i = qVirt; i > pOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.rx(-M_PI_2, qubits[qVirt]);
  kernel.h(qubits[pOcc]);
}

template <typename Kernel>
void doubleExcitationOpt(Kernel &kernel, QuakeValue &qubits, std::size_t pOcc,
                         std::size_t qOcc, std::size_t rVirt, std::size_t sVirt,
                         QuakeValue &theta) {
  std::size_t iOcc, jOcc, aVirt, bVirt;
  double multiplier = 1.;
  if ((pOcc < qOcc) && (rVirt < sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = rVirt;
    bVirt = sVirt;
  } else if ((pOcc > qOcc) && (rVirt > sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = sVirt;
    bVirt = rVirt;
  } else if ((pOcc < qOcc) && (rVirt > sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = sVirt;
    bVirt = rVirt;
    multiplier = -1.;
  } else if ((pOcc > qOcc) && (rVirt < sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = rVirt;
    bVirt = sVirt;
    multiplier = -1.;
  }

  kernel.h(qubits[iOcc]);
  kernel.h(qubits[jOcc]);
  kernel.h(qubits[aVirt]);
  kernel.rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  kernel.rx(-M_PI_2, qubits[bVirt]);
  kernel.h(qubits[aVirt]);

  kernel.rx(M_PI_2, qubits[aVirt]);
  kernel.h(qubits[bVirt]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.rx(-M_PI_2, qubits[aVirt]);
  kernel.h(qubits[jOcc]);

  kernel.rx(M_PI_2, qubits[jOcc]);
  kernel.h(qubits[aVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  kernel.h(qubits[bVirt]);
  kernel.h(qubits[aVirt]);

  kernel.rx(M_PI_2, qubits[aVirt]);
  kernel.rx(M_PI_2, qubits[bVirt]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.rx(-M_PI_2, qubits[jOcc]);
  kernel.h(qubits[iOcc]);

  kernel.rx(M_PI_2, qubits[iOcc]);
  kernel.h(qubits[jOcc]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  kernel.rx(-M_PI_2, qubits[bVirt]);
  kernel.rx(-M_PI_2, qubits[aVirt]);

  kernel.h(qubits[aVirt]);
  kernel.h(qubits[bVirt]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.h(qubits[bVirt]);
  kernel.h(qubits[jOcc]);

  kernel.rx(M_PI_2, qubits[jOcc]);
  kernel.rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  kernel.rx(-M_PI_2, qubits[bVirt]);
  kernel.h(qubits[aVirt]);

  kernel.rx(M_PI_2, qubits[aVirt]);
  kernel.h(qubits[bVirt]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.h(qubits[bVirt]);
  kernel.rx(-M_PI_2, qubits[aVirt]);
  kernel.rx(-M_PI_2, qubits[jOcc]);
  kernel.rx(-M_PI_2, qubits[iOcc]);
}

__qpu__ void doubleExcitationOpt(cudaq::qview<> qubits, std::size_t pOcc,
                                 std::size_t qOcc, std::size_t rVirt,
                                 std::size_t sVirt, double theta) {
  std::size_t iOcc, jOcc, aVirt, bVirt;
  if ((pOcc < qOcc) && (rVirt < sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = rVirt;
    bVirt = sVirt;
  } else if ((pOcc > qOcc) && (rVirt > sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = sVirt;
    bVirt = rVirt;
  } else if ((pOcc < qOcc) && (rVirt > sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = sVirt;
    bVirt = rVirt;
    theta *= -1.;
  } else if ((pOcc > qOcc) && (rVirt < sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = rVirt;
    bVirt = sVirt;
    theta *= -1.;
  }

  h(qubits[iOcc]);
  h(qubits[jOcc]);
  h(qubits[aVirt]);
  rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);

  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[aVirt]);
  h(qubits[jOcc]);

  rx(M_PI_2, qubits[jOcc]);
  h(qubits[aVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  h(qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  rx(M_PI_2, qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[jOcc]);
  h(qubits[iOcc]);

  rx(M_PI_2, qubits[iOcc]);
  h(qubits[jOcc]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  rx(-M_PI_2, qubits[aVirt]);

  h(qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[bVirt]);
  h(qubits[jOcc]);

  rx(M_PI_2, qubits[jOcc]);
  rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[bVirt]);
  rx(-M_PI_2, qubits[aVirt]);
  rx(-M_PI_2, qubits[jOcc]);
  rx(-M_PI_2, qubits[iOcc]);
}

__qpu__ void uccsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
                   std::size_t numElectrons) {

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      get_uccsd_excitations(numElectrons, 0, qubits.size());

  std::size_t thetaCounter = 0;
  for (auto i : cudaq::range(singlesAlpha.size()))
    singleExcitation(qubits, singlesAlpha[i][0], singlesAlpha[i][1],
                     thetas[thetaCounter++]);

  for (auto i : cudaq::range(singlesBeta.size()))
    singleExcitation(qubits, singlesBeta[i][0], singlesBeta[i][1],
                     thetas[thetaCounter++]);

  for (auto i : cudaq::range(doublesMixed.size()))
    doubleExcitationOpt(qubits, doublesMixed[i][0], doublesMixed[i][1],
                        doublesMixed[i][2], doublesMixed[i][3],
                        thetas[thetaCounter++]);

  for (auto i : cudaq::range(doublesAlpha.size()))
    doubleExcitationOpt(qubits, doublesAlpha[i][0], doublesAlpha[i][1],
                        doublesAlpha[i][2], doublesAlpha[i][3],
                        thetas[thetaCounter++]);

  for (auto i : cudaq::range(doublesBeta.size()))
    doubleExcitationOpt(qubits, doublesBeta[i][0], doublesBeta[i][1],
                        doublesBeta[i][2], doublesBeta[i][3],
                        thetas[thetaCounter++]);
}

template <typename Kernel>
void uccsd(Kernel &kernel, QuakeValue &qubits, QuakeValue &thetas,
           std::size_t numElectrons, std::size_t numQubits) {

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      get_uccsd_excitations(numElectrons, 0, numQubits);

  std::size_t thetaCounter = 0;
  for (auto i : cudaq::range(singlesAlpha.size())) {
    // FIXME fix const correctness on quake value
    auto theta = thetas[thetaCounter++];
    singleExcitation(kernel, qubits, singlesAlpha[i][0], singlesAlpha[i][1],
                     theta);
  }

  for (auto i : cudaq::range(singlesBeta.size())) {
    auto theta = thetas[thetaCounter++];
    singleExcitation(kernel, qubits, singlesBeta[i][0], singlesBeta[i][1],
                     theta);
  }

  for (auto i : cudaq::range(doublesMixed.size())) {
    auto theta = thetas[thetaCounter++];
    doubleExcitationOpt(kernel, qubits, doublesMixed[i][0], doublesMixed[i][1],
                        doublesMixed[i][2], doublesMixed[i][3], theta);
  }

  for (auto i : cudaq::range(doublesAlpha.size())) {
    auto theta = thetas[thetaCounter++];
    doubleExcitationOpt(kernel, qubits, doublesAlpha[i][0], doublesAlpha[i][1],
                        doublesAlpha[i][2], doublesAlpha[i][3], theta);
  }

  for (auto i : cudaq::range(doublesBeta.size())) {
    auto theta = thetas[thetaCounter++];
    doubleExcitationOpt(kernel, qubits, doublesBeta[i][0], doublesBeta[i][1],
                        doublesBeta[i][2], doublesBeta[i][3], theta);
  }
}
} // namespace cudaq