# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from ..hamiltonian_generator import HamiltonianGenerator
import numpy as np
import json
from functools import reduce
try:
    from pyscf import gto, scf, cc, ao2mo, mp, mcscf, solvent, fci
except ValueError:
    print(
        'PySCF should be installed to use cudaq-pyscf tool. Use pip install pyscf'
    )

from .utils.cppe_wrapper import PolEmbed


class PolarizableEmbeddedGenerator(HamiltonianGenerator):

    def name(self):
        return 'polarizable_embedded'

    def generate_molecular_spin_ham_restricted(self, h1e, h2e, ecore):

        # This function generates the molecular spin Hamiltonian
        # H= E_core+sum_{pq}  h_{pq} a_p^dagger a_q +
        #                          0.5 * h_{pqrs} a_p^dagger a_q^dagger a_r a_s
        # h1e: one body integrals h_{pq}
        # h2e: two body integrals h_{pqrs}
        # ecore: constant (nuclear repulsion or core energy in the active space Hamiltonian)

        # Total number of qubits equals the number of spin molecular orbitals
        nqubits = 2 * h1e.shape[0]

        # Initialization
        one_body_coeff = np.zeros((nqubits, nqubits))
        two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))

        for p in range(nqubits // 2):
            for q in range(nqubits // 2):

                # p & q have the same spin <a|a>= <b|b>=1
                # <a|b>=<b|a>=0 (orthogonal)
                one_body_coeff[2 * p, 2 * q] = h1e[p, q]
                one_body_coeff[2 * p + 1, 2 * q + 1] = h1e[p, q]

                for r in range(nqubits // 2):
                    for s in range(nqubits // 2):

                        # Same spin (aaaa, bbbbb) <a|a><a|a>, <b|b><b|b>
                        two_body_coeff[2 * p, 2 * q, 2 * r,
                                       2 * s] = 0.5 * h2e[p, q, r, s]
                        two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                       2 * s + 1] = 0.5 * h2e[p, q, r, s]

                        # Mixed spin(abab, baba) <a|a><b|b>, <b|b><a|a>
                        #<a|b>= 0 (orthogoanl)
                        two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                       2 * s] = 0.5 * h2e[p, q, r, s]
                        two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                       2 * s + 1] = 0.5 * h2e[p, q, r, s]

        return one_body_coeff, two_body_coeff, ecore

    def generate_pe_spin_ham_restricted(self, v_pe):

        # Total number of qubits equals the number of spin molecular orbitals
        nqubits = 2 * v_pe.shape[0]

        # Initialization
        spin_pe_op = np.zeros((nqubits, nqubits))

        for p in range(nqubits // 2):
            for q in range(nqubits // 2):

                # p & q have the same spin <a|a>= <b|b>=1
                # <a|b>=<b|a>=0 (orthogonal)
                spin_pe_op[2 * p, 2 * q] = v_pe[p, q]
                spin_pe_op[2 * p + 1, 2 * q + 1] = v_pe[p, q]

        return spin_pe_op

    def get_spin_hamiltonian(self,xyz:str, potfile:str, spin:int, charge: int, basis:str, symmetry:bool=False, memory:float=4000,cycles:int=100,\
                        initguess:str='minao',nele_cas=None, norb_cas=None, MP2:bool=False, natorb:bool=False,casci:bool=False, \
                            ccsd:bool=False,casscf:bool=False, integrals_natorb:bool=False, integrals_casscf:bool=False, verbose:bool=False, cache=True, outFileName=None):

        if spin != 0:
            print(
                'WARN: UHF is not implemented yet for PE model in Cudaq. RHF & ROHF are only supported.'
            )

        ################################
        # Initialize the molecule
        ################################
        filename = xyz.split('.')[0] if outFileName == None else outFileName
        mol = gto.M(atom=xyz,
                    spin=spin,
                    charge=charge,
                    basis=basis,
                    max_memory=memory,
                    symmetry=symmetry,
                    output=filename + '-pyscf.log',
                    verbose=4)

        ##################################
        # Mean field (HF)
        ##################################

        nelec = mol.nelectron
        if verbose:
            print('[Pyscf] Total number of electrons = ', nelec)

        # HF with PE model.
        mf_pe = scf.RHF(mol)
        mf_pe.init_guess = initguess
        mf_pe.chkfile = filename + '-pyscf.chk'
        mf_pe = solvent.PE(mf_pe, potfile).run()
        norb = mf_pe.mo_coeff.shape[1]
        if verbose:
            print('[Pyscf] Total number of orbitals = ', norb)
        if verbose:
            print('[Pyscf] Total HF energy with solvent:', mf_pe.e_tot)
        if verbose:
            print('[Pyscf] Polarizable embedding energy from HF: ',
                  mf_pe.with_solvent.e)

        dm = mf_pe.make_rdm1()

        ##################
        # MP2
        ##################
        if MP2:

            if spin != 0:
                raise ValueError("WARN: ROMP2 is unvailable in pyscf.")
            else:
                mymp = mp.MP2(mf_pe)
                mymp = solvent.PE(mymp, potfile)
                mymp.run()
                if verbose:
                    print('[pyscf] R-MP2 energy with solvent= ', mymp.e_tot)
                if verbose:
                    print('[Pyscf] Polarizable embedding energy from MP: ',
                          mymp.with_solvent.e)

                if integrals_natorb or natorb:
                    # Compute natural orbitals
                    noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
                    if verbose:
                        print(
                            '[Pyscf] Natural orbital occupation number from R-MP2: '
                        )
                    if verbose:
                        print(noons)

        #################
        # CASCI
        #################
        if casci:

            if nele_cas is None:

                #myfci=fci.FCI(mf_pe)
                #myfci=solvent.PE(myfci, args.potfile,dm)
                #myfci.run()
                #if verbose: print('[pyscf] FCI energy with solvent= ', myfci.e_tot)
                #if verbose: print('[Pyscf] Polarizable embedding energy from FCI: ', myfci.with_solvent.e)
                print('[Pyscf] FCI with PE is not supported.')

            else:
                if natorb and (spin == 0):
                    mycasci = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
                    mycasci = solvent.PE(mycasci, potfile)
                    mycasci.run(natorbs)
                    if verbose:
                        print(
                            '[pyscf] CASCI energy (using natural orbitals) with solvent= ',
                            mycasci.e_tot)

                else:
                    mycasci = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
                    mycasci = solvent.PE(mycasci, potfile)
                    mycasci.run()
                    if verbose:
                        print(
                            '[pyscf] CASCI energy (using molecular orbitals) with solvent= ',
                            mycasci.e_tot)
                    if verbose:
                        print(
                            '[Pyscf] Polarizable embedding energy from CASCI: ',
                            mycasci.with_solvent.e)

        #################
        ## CCSD
        #################
        if ccsd:

            if nele_cas is None:
                mycc = cc.CCSD(mf_pe)
                mycc = solvent.PE(mycc, potfile)
                mycc.run()
                if verbose:
                    print('[Pyscf] Total CCSD energy with solvent: ',
                          mycc.e_tot)
                if verbose:
                    print('[Pyscf] Polarizable embedding energy from CCSD: ',
                          mycc.with_solvent.e)

            else:
                mc = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
                frozen = []
                frozen += [y for y in range(0, mc.ncore)]
                frozen += [
                    y for y in range(mc.ncore + norb_cas, len(mf_pe.mo_coeff))
                ]

                if natorb and (spin == 0):
                    mycc = cc.CCSD(mf_pe, frozen=frozen, mo_coeff=natorbs)
                    mycc = solvent.PE(mycc, potfile)
                    mycc.run()
                    if verbose:
                        print(
                            '[pyscf] R-CCSD energy of the active space (using natural orbitals) with solvent= ',
                            mycc.e_tot)
                else:
                    mycc = cc.CCSD(mf_pe, frozen=frozen)
                    mycc = solvent.PE(mycc, potfile)
                    mycc.run()
                    if verbose:
                        print(
                            '[pyscf] CCSD energy of the active space (using molecular orbitals) with solvent= ',
                            mycc.e_tot)
                    if verbose:
                        print(
                            '[Pyscf] Polarizable embedding energy from CCSD: ',
                            mycc.with_solvent.e)
        ############################
        # CASSCF
        ############################
        if casscf:
            if natorb and (spin == 0):
                mycas = mcscf.CASSCF(mf_pe, norb_cas, nele_cas)
                mycas = solvent.PE(mycas, potfile)
                mycas.max_cycle_macro = cycles
                mycas.kernel(natorbs)
                if verbose:
                    print(
                        '[pyscf] CASSCF energy (using natural orbitals) with solvent= ',
                        mycas.e_tot)

            else:
                mycas = mcscf.CASSCF(mf_pe, norb_cas, nele_cas)
                mycas = solvent.PE(mycas, potfile)
                mycas.max_cycle_macro = cycles
                mycas.kernel()
                if verbose:
                    print(
                        '[pyscf] CASSCF energy (using molecular orbitals) with solvent= ',
                        mycas.e_tot)

        ###########################################################################
        # Computation of one and two electron integrals for the QC+PE
        ###########################################################################
        if nele_cas is None:
            # Compute the 1e integral in atomic orbital then convert to HF basis
            h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
            ## Ways to convert from ao to mo
            #h1e=np.einsum('pi,pq,qj->ij', myhf.mo_coeff, h1e_ao, myhf.mo_coeff)
            h1e = reduce(np.dot, (mf_pe.mo_coeff.T, h1e_ao, mf_pe.mo_coeff))
            #h1e=reduce(np.dot, (myhf.mo_coeff.conj().T, h1e_ao, myhf.mo_coeff))

            # Compute the 2e integrals then convert to HF basis
            h2e_ao = mol.intor("int2e_sph", aosym='1')
            h2e = ao2mo.incore.full(h2e_ao, mf_pe.mo_coeff)

            # Reorder the chemist notation (pq|rs) ERI h_prqs to h_pqrs
            # a_p^dagger a_r a_q^dagger a_s --> a_p^dagger a_q^dagger a_r a_s
            h2e = h2e.transpose(0, 2, 3, 1)

            nuclear_repulsion = mf_pe.energy_nuc()

            # Compute the molecular spin electronic Hamiltonian from the
            # molecular electron integrals
            obi, tbi, e_nn = self.generate_molecular_spin_ham_restricted(
                h1e, h2e, nuclear_repulsion)

            # Dump obi and tbi to binary file.
            if cache:
                obi.astype(complex).tofile(f'{filename}_one_body.dat')
                tbi.astype(complex).tofile(f'{filename}_two_body.dat')

            # Compute the PE contribution to the Hamiltonian
            dm = mf_pe.make_rdm1()

            mype = PolEmbed(mol, potfile)
            E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)

            # convert V_pe from atomic orbital to molecular orbital representation
            V_pe_mo = reduce(np.dot, (mf_pe.mo_coeff.T, V_pe, mf_pe.mo_coeff))

            obi_pe = self.generate_pe_spin_ham_restricted(V_pe_mo)
            if cache:
                obi_pe.astype(complex).tofile(f'{filename}_pe_one_body.dat')

            metadata = {
                'num_electrons': nelec,
                'num_orbitals': norb,
                'nuclear_energy': e_nn,
                'PE_energy': E_pe,
                'HF_energy': mf_pe.e_tot
            }
            if cache:
                with open(f'{filename}_metadata.json', 'w') as f:
                    json.dump(metadata, f)

            return (obi, tbi, nuclear_repulsion, obi_pe, nelec, norb)

        else:
            if integrals_natorb:
                mc = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
                h1e_cas, ecore = mc.get_h1eff(natorbs)
                h2e_cas = mc.get_h2eff(natorbs)
                h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')

                obi, tbi, core_energy = self.generate_molecular_spin_ham_restricted(
                    h1e_cas, h2e_cas, ecore)

                # Dump obi and tbi to binary file.
                if cache:
                    obi.astype(complex).tofile(f'{filename}_one_body.dat')
                    tbi.astype(complex).tofile(f'{filename}_two_body.dat')

                if casci:

                    dm = mcscf.make_rdm1(mycasci)
                    mype = PolEmbed(mol, potfile)
                    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
                    #convert from ao to mo

                    #V_pe_mo=reduce(np.dot, (mf_pe.mo_coeff.T, V_pe, mf_pe.mo_coeff))
                    V_pe_mo = reduce(np.dot, (natorbs.T, V_pe, natorbs))

                    V_pe_cas = V_pe_mo[mycasci.ncore:mycasci.ncore +
                                       mycasci.ncas,
                                       mycasci.ncore:mycasci.ncore +
                                       mycasci.ncas]

                    obi_pe = self.generate_pe_spin_ham_restricted(V_pe_cas)
                    if cache:
                        obi_pe.astype(complex).tofile(
                            f'{filename}_pe_one_body.dat')

                    metadata = {
                        'num_electrons': nele_cas,
                        'num_orbitals': norb_cas,
                        'core_energy': ecore,
                        'PE_energy': E_pe,
                        'HF_energy': mf_pe.e_tot
                    }
                    if cache:
                        with open(f'{filename}_metadata.json', 'w') as f:
                            json.dump(metadata, f)

                    return (obi, tbi, ecore, obi_pe, nele_cas, norb_cas)

                else:
                    raise ValueError('You should use casci=True.')

            elif integrals_casscf:
                if casscf:
                    h1e_cas, ecore = mycas.get_h1eff()
                    h2e_cas = mycas.get_h2eff()
                    h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                    h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1),
                                         order='C')
                else:
                    raise ValueError(
                        "WARN: You need to run casscf. Use casscf=True.")
                obi, tbi, core_energy = self.generate_molecular_spin_ham_restricted(
                    h1e_cas, h2e_cas, ecore)

                # Dump obi and tbi to binary file.
                if cache:
                    obi.astype(complex).tofile(f'{filename}_one_body.dat')
                    tbi.astype(complex).tofile(f'{filename}_two_body.dat')

                dm = mcscf.make_rdm1(mycas)
                # Compute the PE contribution to the Hamiltonian
                mype = PolEmbed(mol, potfile)
                E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
                #convert from ao to mo
                V_pe_mo = reduce(np.dot,
                                 (mycas.mo_coeff.T, V_pe, mycas.mo_coeff))

                V_pe_cas = V_pe_mo[mycas.ncore:mycas.ncore + mycas.ncas,
                                   mycas.ncore:mycas.ncore + mycas.ncas]
                obi_pe = self.generate_pe_spin_ham_restricted(V_pe_cas)
                if cache:
                    obi_pe.astype(complex).tofile(f'{filename}_pe_one_body.dat')

                metadata = {
                    'num_electrons': nele_cas,
                    'num_orbitals': norb_cas,
                    'core_energy': ecore,
                    'PE_energy': E_pe,
                    'HF_energy': mf_pe.e_tot
                }
                if cache:
                    with open(f'{filename}_metadata.json', 'w') as f:
                        json.dump(metadata, f)

                return (obi, tbi, ecore, obi_pe, nele_cas, norb_cas)

            else:
                mc = mcscf.CASCI(mf_pe, norb_cas, nele_cas)
                h1e_cas, ecore = mc.get_h1eff(mf_pe.mo_coeff)
                h2e_cas = mc.get_h2eff(mf_pe.mo_coeff)
                h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')
                obi, tbi, core_energy = self.generate_molecular_spin_ham_restricted(
                    h1e_cas, h2e_cas, ecore)

                # Dump obi and tbi to binary file.
                if cache:
                    obi.astype(complex).tofile(f'{filename}_one_body.dat')
                    tbi.astype(complex).tofile(f'{filename}_two_body.dat')

                dm = mf_pe.make_rdm1()
                # Compute the PE contribution to the Hamiltonian
                mype = PolEmbed(mol, potfile)
                E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
                #convert from ao to mo
                V_pe_mo = reduce(np.dot,
                                 (mf_pe.mo_coeff.T, V_pe, mf_pe.mo_coeff))

                V_pe_cas = V_pe_mo[mc.ncore:mc.ncore + mc.ncas,
                                   mc.ncore:mc.ncore + mc.ncas]
                obi_pe = self.generate_pe_spin_ham_restricted(V_pe_cas)
                if cache:
                    obi_pe.astype(complex).tofile(f'{filename}_pe_one_body.dat')

                metadata = {
                    'num_electrons': nele_cas,
                    'num_orbitals': norb_cas,
                    'core_energy': ecore,
                    'PE_energy': E_pe,
                    'HF_energy': mf_pe.e_tot
                }
                if cache:
                    with open(f'{filename}_metadata.json', 'w') as f:
                        json.dump(metadata, f)

                return (obi, tbi, ecore, obi_pe, nele_cas, norb_cas)

    def generate(self, xyz, basis, **kwargs):
        requiredOptions = ['potfile', 'spin', 'charge']
        for option in requiredOptions:
            if option not in kwargs:
                raise RuntimeError(
                    f'solvent Hamiltonian generator missing required option - {option}'
                )

        potfile = kwargs['potfile']
        spin = kwargs['spin']
        charge = kwargs['charge']
        symmetry = kwargs['symmetry'] if 'symmetry' in kwargs else False
        memory = kwargs['memory'] if 'memory' in kwargs else 4000
        cycles = kwargs['cycles'] if 'cycles' in kwargs else 100
        initguess = kwargs['initguess'] if 'initguess' in kwargs else 'minao'
        nele_cas = kwargs['nele_cas'] if 'nele_cas' in kwargs else None
        norb_cas = kwargs['norb_cas'] if 'norb_cas' in kwargs else None
        MP2 = kwargs['MP2'] if 'MP2' in kwargs else False
        natorb = kwargs['natorb'] if 'natorb' in kwargs else False
        casci = kwargs['casci'] if 'casci' in kwargs else False
        ccsd = kwargs['ccsd'] if 'ccsd' in kwargs else False
        casscf = kwargs['casscf'] if 'casscf' in kwargs else False
        integrals_natorb = kwargs[
            'integrals_natorb'] if 'integrals_natorb' in kwargs else False
        integrals_casscf = kwargs[
            'integrals_casscf'] if 'integrals_casscf' in kwargs else False
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        cache_data = kwargs['cache'] if 'cache' in kwargs else True
        outfilename = kwargs[
            'out_file_name'] if 'out_file_name' in kwargs else None
        return self.get_spin_hamiltonian(xyz, potfile, spin, charge, basis,
                                         symmetry, memory, cycles, initguess,
                                         nele_cas, norb_cas, MP2, natorb, casci,
                                         ccsd, casscf, integrals_natorb,
                                         integrals_casscf, verbose, cache_data,
                                         outfilename)


def get_hamiltonian_generator():
    return PolarizableEmbeddedGenerator()
