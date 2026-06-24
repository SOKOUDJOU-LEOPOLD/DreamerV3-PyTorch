#!/bin/bash
#SBATCH --job-name=export_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/lsokoudj/DreamerV3-PyTorch/jobs/logs/export_diagnostics.out
#SBATCH --mail-user=lsokoudj@tamu.edu
#SBATCH --mail-type=ALL

source /scratch/user/lsokoudj/DreamerV3-PyTorch/dreamer_env/bin/activate
cd /scratch/user/lsokoudj/DreamerV3-PyTorch

BASE_CFG='{"model_size": "L", "epochs": 3}'
ABL1_CFG='{"model_size": "L", "ablation_no_symlog_twohot": true, "epochs": 10}'
ABL2_CFG='{"model_size": "L", "free_nats": 0.0, "epochs": 10}'
ABL3_CFG='{"model_size": "L", "return_norm_method": "std", "epochs": 10}'
ABL4_CFG='{"model_size": "L", "loss_decoder_scale": 0.0, "epochs": 10}'

# env_name|run_name|override_config|checkpoint
RUNS=(
  "car_racing|baseline|$BASE_CFG|checkpoints_epoch_2_step_5000.ckpt"
  "car_racing|baseline|$BASE_CFG|checkpoints_epoch_3_step_10000.ckpt"
  "car_racing|baseline|$BASE_CFG|checkpoints_epoch_3_step_14987.ckpt"
  "atari100k-breakout|baseline|$BASE_CFG|checkpoints_epoch_1_step_2000.ckpt"
  "atari100k-breakout|baseline|$BASE_CFG|checkpoints_epoch_2_step_4000.ckpt"
  "atari100k-breakout|baseline|$BASE_CFG|checkpoints_epoch_3_step_6000.ckpt"
  "car_racing|ablation1_no_symlog|$ABL1_CFG|checkpoints_epoch_2_step_6500.ckpt"
  "atari100k-breakout|ablation1_no_symlog|$ABL1_CFG|checkpoints_epoch_4_step_6500.ckpt"
  "car_racing|ablation2_no_freebits|$ABL2_CFG|checkpoints_epoch_1_step_500.ckpt"
  "atari100k-breakout|ablation2_no_freebits|$ABL2_CFG|checkpoints_epoch_1_step_500.ckpt"
  "car_racing|ablation3_std_norm|$ABL3_CFG|checkpoints_epoch_2_step_9500.ckpt"
  "atari100k-breakout|ablation3_std_norm|$ABL3_CFG|checkpoints_epoch_1_step_1500.ckpt"
  "car_racing|ablation4_no_decoder|$ABL4_CFG|checkpoints_epoch_3_step_11500.ckpt"
  "atari100k-breakout|ablation4_no_decoder|$ABL4_CFG|checkpoints_epoch_4_step_6500.ckpt"
)

for entry in "${RUNS[@]}"; do
  IFS='|' read -r env_name run_name override_config checkpoint <<< "$entry"
  echo "=============================================="
  echo "env=$env_name run=$run_name ckpt=$checkpoint"
  echo "=============================================="
  python3 tools/export_diagnostic_videos.py \
    --env_name "$env_name" \
    --run_name "$run_name" \
    --override_config "$override_config" \
    --checkpoint "$checkpoint" \
    --num_clips 1
  echo "EXIT_CODE=$?"
done
