#!/bin/bash
#
#SBATCH --job-name=gtn
#SBATCH --partition=ivm
#SBATCH --nodelist node242
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00
#SBATCH --mem-per-cpu=16000
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bendickens@zoho.com

source activate snk
python3 /scistor/ivm/bds550/snkit/src/snkit/controller.py

