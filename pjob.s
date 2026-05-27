#BSUB -J ringfit
#BSUB -q astron1
#BSUB -n 30
#BSUB -R "span[hosts=1]"
#BSUB -o %J.out
#BSUB -e %J.err

source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mpiexec -np 30 python -m dring fit -c configs/HD163296_ring1.yaml
