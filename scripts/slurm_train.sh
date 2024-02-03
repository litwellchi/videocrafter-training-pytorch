#!/bin/bash
#SBATCH --job-name=debug_2node
#SBATCH --output=torch_lightning_output.txt
#SBATCH --error=torch_lightning_error.txt
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --nodelist=dgx-124,dgx-123
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
echo SLURM_PROCID: $SLURM_PROCID
export LOGLEVEL=INFO

export PYTHONFAULTHANDLER=1

export NVIDIA_VISIBLE_DEVICES=all \
export NCCL_SOCKET_IFNAME=ibp \
export NCCL_IB_HCA=mlx5 \
# export NCCL_DEBUG=INFO \
export NCCL_DEBUG=ERROR \
# export NCCL_DEBUG_SUBSYS=ALL \
export GPUS_PER_NODE=8 \
export MASTER_ADDR=$(echo $head_node_ip) \
export MASTER_PORT=6000 \
export NODE_RANK=$SLURM_PROCID \
export NNODES=2 \
export CUDA_DEVICE_MAX_CONNECTIONS=10 \
export OMP_NUM_THREADS=10 \

PROJ_ROOT="./"                      # root directory for saving experiment logs
EXPNAME="overfit_macvid"          # experiment name 
DATADIR="/aifs4su/mmdata/rawdata/videogen/macvid/video_dataset_85"  # dataset directory
CONFIG="configs/train_t2v_1024_v1.0.yaml"
CKPT_RESUME="/aifs4su/mmcode/videogen/share_ckpts/VideoCrafter/Text2Video-1024/model.ckpt"
# run
export TOKENIZERS_PARALLELISM=false
# export 

# conda activate videocrafter
srun python train_main.py \
--base $CONFIG \
-t --gpus '0,1,2,3,4,5,6,7', \
--name $EXPNAME \
--logdir $PROJ_ROOT \
--auto_resume True \
lightning.trainer.num_nodes=2 \
--load_from_checkpoint $CKPT_RESUME 
# srun scripts/train_vc.sh