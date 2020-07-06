!/bin/bash
#SBATCH --job-name=t1
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/bds550/gtn/output/example.csv
#SBATCH --nodelist node241
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1000
