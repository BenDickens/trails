#!/bin/bash
#SBATCH --job-name=gtn
#SBATCH --partition=ivm
#SBATCH --nodelist node240
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1000

source activate snk
python3 /scistor/ivm/bds550/snkit/src/snkit/controller2.py

