#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Example usage:
# python create_experiment.py --pdb 6e5b -FPGB

from jinja2 import Environment, FileSystemLoader

import os
import argparse

parser = argparse.ArgumentParser(description="Generate experiment scripts from templates.")
parser.add_argument("--pdb", required=True, help="Value for PDB.")
parser.add_argument("-F", action='store_true', default=False, help="Value for USES_PHOTON_FLUCTUATION.")
parser.add_argument("-P", action='store_true', default=False, help="Value for USES_POISSON_NOISE.")
parser.add_argument("-G", action='store_true', default=False, help="Value for USES_GAUSSIAN_NOISE.")
parser.add_argument("-B", action='store_true', default=False, help="Value for USES_BEAM_STOP_MASK.")
parser.add_argument("-C", action='store_true', default=False, help="Value for USES_RANDOM_PATCH.")
args = parser.parse_args()

pdb = args.pdb
F   = args.F
P   = args.P
G   = args.G
B   = args.B
C   = args.C

# Set up the environment and the loader
env = Environment(loader=FileSystemLoader('.'))
dir_bash = 'bash'
dir_yaml = 'yaml'
bash_template = env.get_template('bash/template.sh')
yaml_template = env.get_template('yaml/template.yaml')

# Render the yaml script
yaml_content = yaml_template.render(pdb=pdb.upper(), F=F, P=P, G=G, B=B, C=C)

mode = {
    "f" : F,
    "p" : P,
    "g" : G,
    "b" : B,
    "c" : C,
}
mode_encode = ''.join([ k for k, v in mode.items() if v ])
basename    = f"{pdb}_resnet18_coslr_{mode_encode}"

fl_yaml   = f"{basename}.yaml"
path_yaml = os.path.join(dir_yaml, fl_yaml)
with open(path_yaml, 'w') as fh:
    fh.write(yaml_content)

# Render the bash script
bash_content = bash_template.render(path_yaml = path_yaml)

fl_bash   = f"{basename}.sh"
path_bash = os.path.join(dir_bash, fl_bash)
with open(path_bash, 'w') as fh:
    fh.write(bash_content)

