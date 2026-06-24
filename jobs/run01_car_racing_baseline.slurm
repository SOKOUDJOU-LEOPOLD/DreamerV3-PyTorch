#!/bin/bash
#SBATCH --job-name=cr_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/user/lsokoudj/DreamerV3-PyTorch/jobs/logs/run01_cr_base.out
#SBATCH --mail-user=lsokoudj@tamu.edu
#SBATCH --mail-type=ALL

source /scratch/user/lsokoudj/DreamerV3-PyTorch/dreamer_env/bin/activate
cd /scratch/user/lsokoudj/DreamerV3-PyTorch

env_name=car_racing run_name=baseline override_config='{"model_size": "L", "epochs": 3}' \
  python3 main.py -c configs/DreamerV3/dreamer_v3.py -m training \
  --saving_period_step 500 --log_figure_period_step 500

echo "EXIT_CODE=$?"
