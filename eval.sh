#!/bin/bash
#SBATCH --job-name=av2_eval
#SBATCH --partition=ada24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

alias smem=". /srv/cluster/bin/smem"

hostname

source ~/.bashrc
conda init
conda activate icpflow
which python

export CUDA_HOME=/srv/cluster/local/cuda-12.1
export PATH=$CUDA_HOME/bin/:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CPATH=~/miniconda3/envs/icpflow/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pczyl2/miniconda3/envs/icpflow/lib

which nvcc
nvcc --version
echo $CUDA_VISIBLE_DEVICE
nvidia-smi --query-gpu=index,name --format=csv,noheader

# cd /home/pczyl2/projects/SceneFlow/VoteFlowpp/assets/cuda/chamfer3D
# python ./setup.py install

# cd /home/pczyl2/projects/SceneFlow/VoteFlowpp/assets/cuda/nn_pillar
# python ./setup.py install

# cd /home/pczyl2/projects/SceneFlow/VoteFlowpp/assets/cuda/mmvc
# python ./setup.py install

cd /home/pczyl2/projects/SceneFlow/OpenSceneFlow
python eval.py model=icpflowpp