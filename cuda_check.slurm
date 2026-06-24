#!/bin/bash
#SBATCH --job-name=cuda_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/user/lsokoudj/DreamerV3-PyTorch/cuda_check.out

source /scratch/user/lsokoudj/DreamerV3-PyTorch/dreamer_env/bin/activate
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.zeros(1).cuda())"
