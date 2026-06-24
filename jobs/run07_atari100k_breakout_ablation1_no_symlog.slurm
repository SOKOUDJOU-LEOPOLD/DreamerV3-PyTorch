#!/bin/bash
#SBATCH --job-name=at_abl1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/lsokoudj/DreamerV3-PyTorch/jobs/logs/run07_at_abl1.out
#SBATCH --mail-user=lsokoudj@tamu.edu
#SBATCH --mail-type=ALL

source /scratch/user/lsokoudj/DreamerV3-PyTorch/dreamer_env/bin/activate
cd /scratch/user/lsokoudj/DreamerV3-PyTorch

env_name=atari100k-breakout run_name=ablation1_no_symlog override_config='{"model_size": "L", "ablation_no_symlog_twohot": true, "epochs": 10}' \
  python3 main.py -c configs/DreamerV3/dreamer_v3.py -m training \
  --saving_period_step 500 --log_figure_period_step 500

echo "EXIT_CODE=$?"
