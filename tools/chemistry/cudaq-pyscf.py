#!/usr/bin/env python3

import argparse, json
import importlib, pkgutil
import cudaqlib.tools.chemistry.pyscf.generators


def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


discovered_plugins = {}
for finder, name, ispkg in iter_namespace(cudaqlib.tools.chemistry.pyscf.generators):
    try:
        discovered_plugins[name] = importlib.import_module(name)
    except ModuleNotFoundError:
        pass

hamiltonianGenerators = {
    plugin.get_hamiltonian_generator().name(): plugin
    for _, plugin in discovered_plugins.items()
}

#############################
# Argument Parser
#############################

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument(
    '--type',
    type=str,
    help='type of simulation (hamiltonian generator) - options include {}'.
    format([k for k, v in hamiltonianGenerators.items()]),
    default='gas_phase')
parser.add_argument('--xyz', help="xyz file", type=str)
parser.add_argument('--basis', help='', type=str)
parser.add_argument('--charge', help="charge of the system", type=int)
parser.add_argument('--out-file-name',
                    help='base file name for output data.',
                    type=str)
parser.add_argument('--spin', help="no. of unpaired electrons (2 *s)", type=int)
parser.add_argument('--symmetry', help="", action='store_true', default=False)
parser.add_argument('--memory', help="", type=float, default=4000)
parser.add_argument('--cycles', help="", type=int, default=100)
parser.add_argument('--initguess', help="", type=str, default='minao')
parser.add_argument('--UR', help="", action='store_true', default=False)
parser.add_argument('--MP2', help="", action='store_true', default=False)
parser.add_argument('--nele_cas', help="", type=int, default=None)
parser.add_argument('--norb_cas', help="", type=int, default=None)
parser.add_argument('--natorb', help="", action='store_true', default=False)
parser.add_argument('--casci', help="", action='store_true', default=False)
parser.add_argument('--ccsd', help="", action='store_true', default=False)
parser.add_argument('--casscf', help="", action='store_true', default=False)
parser.add_argument('--integrals_natorb',
                    help="",
                    action='store_true',
                    default=False)
parser.add_argument('--integrals_casscf',
                    help="",
                    action='store_true',
                    default=False)
parser.add_argument('--potfile', help="", type=str, default=None)
parser.add_argument('--verbose',
                    help="Verbose printout",
                    action='store_true',
                    default=False)

# Parse the arguments
args = parser.parse_args()
if args.type not in hamiltonianGenerators:
    raise RuntimeError(f'invalid hamiltonian generator type - {args.type}')
hamiltonianGenerator = hamiltonianGenerators[
    args.type].get_hamiltonian_generator()

filterArgs = ['xyz', 'basis']
filteredArgs = {k: v for (k, v) in vars(args).items() if k not in filterArgs}
hamiltonianGenerator.generate(args.xyz, args.basis, **filteredArgs)
