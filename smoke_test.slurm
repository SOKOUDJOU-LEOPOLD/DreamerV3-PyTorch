#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/user/lsokoudj/DreamerV3-PyTorch/smoke_test.out

source /scratch/user/lsokoudj/DreamerV3-PyTorch/dreamer_env/bin/activate
cd /scratch/user/lsokoudj/DreamerV3-PyTorch

echo "=== car_racing smoke test (real training, GPU, L model) ==="
env_name=car_racing run_name=smoke_test override_config='{"model_size":"L","epochs":1}' \
  timeout 300 python3 main.py -c configs/DreamerV3/dreamer_v3.py -m training --steps_per_epoch 30 --step_log_period 5 --saving_period_epoch 1000 --log_figure_period_epoch 1000 --no_eval_training
echo "EXIT_CODE_CAR_RACING=$?"

echo "=== atari100k-breakout smoke test (real training, GPU, L model) ==="
env_name=atari100k-breakout run_name=smoke_test override_config='{"model_size":"L","epochs":1}' \
  timeout 300 python3 main.py -c configs/DreamerV3/dreamer_v3.py -m training --steps_per_epoch 30 --step_log_period 5 --saving_period_epoch 1000 --log_figure_period_epoch 1000 --no_eval_training
echo "EXIT_CODE_ATARI=$?"
