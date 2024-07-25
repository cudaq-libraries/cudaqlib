import os.path, sys, os
import numpy as np
import pathlib
import cudaqlib 
currentPath = pathlib.Path(__file__).parent.resolve()
from cudaqlib.tools.chemistry.pyscf.generators.gas_phase_generator import GasPhaseGenerator
from cudaqlib.tools.chemistry.pyscf.generators.polarizable_embedded_generator import PolarizableEmbeddedGenerator

gas_gen = GasPhaseGenerator()
pe_gen = PolarizableEmbeddedGenerator()

# FIXME when no cppe is installed

def test_gas_phase():
    print('hello: ', currentPath)
    data = gas_gen.generate(str(currentPath) + '/resources/LiH.xyz',
                            'sto3g',
                            spin=0,
                            charge=0,
                            MP2=True,
                            casci=True,
                            ccsd=True,
                            verbose=True,
                            cache=False)
    print(data['num_orbitals'], data['num_electrons'])
    assert np.isclose(1.058354, data['energies']['nuclear_energy'], rtol=1e-6)
    assert data['num_electrons'] == 4
    assert data['num_orbitals'] == 6


def test_gas_phase_active_space():

    data = gas_gen.generate(xyz=str(currentPath) + '/resources/N2.xyz',
                            spin=0,
                            charge=0,
                            basis='sto3g',
                            nele_cas=6,
                            norb_cas=6,
                            MP2=True,
                            casci=True,
                            ccsd=True,
                            casscf=True,
                            verbose=True,
                            cache=False)
    print(data['energies'])
    assert np.isclose(-96.32874, data['energies']['core_energy'], rtol=1e-3)
    assert data['num_electrons'] == 6
    assert data['num_orbitals'] == 6


def test_gas_phase_active_space_cudaq():
    geometry = [('N', (0., 0., 0.56)), ('N', (0., 0., -.56))]
    molecule = cudaqlib.operators.create_molecule(geometry,
                                                  'sto3g',
                                                  0,
                                                  0,
                                                  verbose=True,
                                                  nele_cas=6,
                                                  norb_cas=6,
                                                  MP2=True,
                                                  casci=True,
                                                  ccsd=True,
                                                  casscf=True)

    print(molecule)
    print(molecule.energies)
    assert molecule.n_electrons == 6
    assert molecule.n_orbitals == 6 
    assert np.isclose(-107.49999, molecule.energies['hf_energy'], atol=1e-3)


def test_pyscf_pe():

    data = pe_gen.generate(str(currentPath) + '/resources/COH2.xyz',
                           potfile=str(currentPath) + '/resources/water-II.pot',
                           spin=0,
                           charge=0,
                           basis='sto3g',
                           MP2=True,
                           ccsd=True,
                           verbose=True,
                           cache=False)

    print(data['energies'])
    assert np.isclose(31.23082, data['energies']['nuclear_energy'], rtol=1e-6)
    assert data['num_electrons'] == 16
    assert data['num_orbitals'] == 12


def test_pyscf_pe_active_space():

    data = pe_gen.generate(str(currentPath) + '/resources/COH2.xyz',
                           potfile=str(currentPath) + '/resources/water-II.pot',
                           spin=0,
                           charge=0,
                           basis='sto3g',
                           nele_cas=6,
                           norb_cas=6,
                           MP2=True,
                           casci=True,
                           ccsd=True,
                           verbose=True,
                           cache=False)

    print(data['energies'])
    assert np.isclose(-101.419971, data['energies']['core_energy'], rtol=1e-6)
    assert data['num_electrons'] == 6
    assert data['num_orbitals'] == 6


def test_pyscf_cppe():

    from cudaqlib.tools.chemistry.pyscf.generators.utils.cppe_wrapper import PolEmbed

    from pyscf import gto, scf, solvent

    mol = gto.M(atom='''
        C        0.000000    0.000000   -0.542500
        O        0.000000    0.000000    0.677500
        H        0.000000    0.935307   -1.082500
        H        0.000000   -0.935307   -1.082500
            ''',
                basis='sto3g')

    myhf = scf.RHF(mol)
    myhf.kernel()
    nelec = mol.nelectron
    assert nelec == 16
    norb = myhf.mo_coeff.shape[1]
    assert norb == 12

    mf = solvent.PE(myhf, str(currentPath) + '/resources/water-II.pot').run()

    dm = mf.make_rdm1()
    mype = PolEmbed(mol, str(currentPath) + '/resources/water-II.pot')
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)

    assert np.isclose(mf.with_solvent.e, E_pe, rtol=1e-5)
    assert np.isclose(mf.with_solvent.v[0, 1], V_es[0, 1], rtol=1e-5)

    from pyscf import cc

    mycc = cc.CCSD(mf)
    mycc = solvent.PE(mycc, str(currentPath) + '/resources/water-II.pot')
    mycc.run()

    dm = mycc.make_rdm1(ao_repr=True)
    mype = PolEmbed(mol, str(currentPath) + '/resources/water-II.pot')
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
    assert np.isclose(mycc.with_solvent.e, E_pe, rtol=1e-5)


def test_cppe_active_space():

    import numpy as np
    from pyscf import gto, scf, mp, mcscf, solvent
    from cudaqlib.tools.chemistry.pyscf.generators.utils.cppe_wrapper import PolEmbed
    from functools import reduce

    mol = gto.M(atom=str(currentPath) + '/resources/NH3.xyz',
                spin=0,
                charge=0,
                basis='631g',
                output='NH3-test-pyscf.log',
                verbose=4)

    mf_pe = scf.RHF(mol)

    mf_pe = solvent.PE(mf_pe,
                       str(currentPath) + '/resources/46_water.pot').run()
    norb = mf_pe.mo_coeff.shape[1]
    #print('HF')
    # dm of the atomic orbitals
    dm = mf_pe.make_rdm1()
    noon, U = np.linalg.eigh(dm)
    noon = np.flip(noon)
    # noon might be larder than 2 because this is dm_ao
    #print(noon)
    mype = PolEmbed(mol, str(currentPath) + '/resources/46_water.pot')
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)

    assert np.isclose(E_pe, mf_pe.with_solvent.e, rtol=1e-3)

    ovlp = mol.intor_symmetric('int1e_ovlp')
    #convert dm from ao to mo
    dm = np.einsum('pi,pq,qj->ij', ovlp, dm, ovlp)
    dm_mo = reduce(np.dot, (mf_pe.mo_coeff.T, dm, mf_pe.mo_coeff))
    mype = PolEmbed(mol, str(currentPath) + '/resources/46_water.pot')
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm_mo)
    assert np.isclose(E_pe, mf_pe.with_solvent.e, rtol=1e-2)

    mymp = mp.MP2(mf_pe)
    mymp = solvent.PE(mymp, str(currentPath) + '/resources/46_water.pot')
    mymp.run()
    noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
    #print(noons)

    #print('CASSCF result')
    mycas = mcscf.CASSCF(mf_pe, 6, 6)
    mycas = solvent.PE(mycas, str(currentPath) + '/resources/46_water.pot')
    mycas.kernel(natorbs)

    dm = mcscf.make_rdm1(mycas)
    noon, U = np.linalg.eigh(dm)
    noon = np.flip(noon)
    #print(noon)
    mype = PolEmbed(mol, str(currentPath) + '/resources/46_water.pot')
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
    assert np.isclose(E_pe, mycas.with_solvent.e, rtol=1e-3)

    ovlp = mol.intor_symmetric('int1e_ovlp')
    dm = np.einsum('pi,pq,qj->ij', ovlp, dm, ovlp)
    dm_mo = reduce(np.dot, (mf_pe.mo_coeff.T, dm, mf_pe.mo_coeff))
    mype = PolEmbed(mol, str(currentPath) + '/resources/46_water.pot')
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm_mo)
    assert np.isclose(E_pe, mycas.with_solvent.e, 1e-2)

    noon, U = np.linalg.eigh(dm_mo)
    noon = np.flip(noon)
    #print('natocc', noon)

    #print('Casci result')
    mycasci = mcscf.CASCI(mf_pe, 6, 6)
    mycasci = solvent.PE(mycasci, str(currentPath) + '/resources/46_water.pot')
    mycasci.run(natorbs)
    dm = mcscf.make_rdm1(mycasci)
    noon, U = np.linalg.eigh(dm)
    noon = np.flip(noon)
    #print(noon)
    mype = PolEmbed(mol, str(currentPath) + '/resources/46_water.pot')
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm)
    assert np.isclose(E_pe, mycas.with_solvent.e, rtol=1e-3)

    dm = np.einsum('pi,pq,qj->ij', ovlp, dm, ovlp)
    dm_mo = reduce(np.dot, (mf_pe.mo_coeff.T, dm, mf_pe.mo_coeff))
    mype = PolEmbed(mol, str(currentPath) + '/resources/46_water.pot')
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm_mo)

    assert np.isclose(E_pe, mycasci.with_solvent.e, 1e-2)

    noon, U = np.linalg.eigh(dm_mo)
    noon = np.flip(noon)
