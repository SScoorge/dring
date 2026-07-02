#BSUB -J ringfit
#BSUB -q astron1
#BSUB -n 30
#BSUB -R "span[hosts=1]"
#BSUB -o %J.out
#BSUB -e %J.err

source .venv/bin/activate

# Optional: keep each MPI rank single-threaded so BLAS/OpenMP libraries do not
# oversubscribe the node. Recommended for multi-rank cluster runs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mpiexec -np 30 python -m dring fit -c configs/mock_sim1.yaml
