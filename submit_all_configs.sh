#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/share/home/yanglinhan/project_papper2/dringtst"
QUEUE="astron1"
NPROC=30

cd "$PROJECT_DIR"
mkdir -p jobs logs

configs=(
  configs/HD163296_ring1.yaml
  configs/HD163296_ring2.yaml
  configs/LKCA15_ring1.yaml
  configs/LKCA15_ring2.yaml
  configs/mock_sim1.yaml
  configs/mock_sim2.yaml
  configs/mock_sim3.yaml
  configs/mock_sim4.yaml
)

for config in "${configs[@]}"; do
  name="$(basename "$config" .yaml)"
  job_script="jobs/${name}.s"

  cat > "$job_script" <<EOF
#BSUB -J dring_${name}
#BSUB -q ${QUEUE}
#BSUB -n ${NPROC}
#BSUB -R "span[hosts=1]"
#BSUB -o logs/${name}.%J.out
#BSUB -e logs/${name}.%J.err

cd ${PROJECT_DIR}
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m dring fit -c ${config} --check-config
mpiexec -np ${NPROC} python -m dring fit -c ${config}
EOF

  echo "Submitting ${job_script}"
  bsub < "$job_script"
done
