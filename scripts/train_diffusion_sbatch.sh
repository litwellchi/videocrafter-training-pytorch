#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBTACH --job-name=cnn_unet_video_16node
#SBATCH --nodes=8            # This needs to match Trainer(num_nodes=...)
#SBATCH -p project   #important and necessary
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=24:00:00 # must set the training time by default. 24h max...
#SBATCH --cpus-per-task=8
#SBATCH --output=srun_output/_%j/output.txt
#SBATCH --error=srun_output/_%j/error.txt
#SBATCH --signal=SIGUSR1@90 # reboot if the process is killed..

# debugging flags (optional)

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo "numnodes:"$nodes

nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export WANDB_API_KEY="231c840bf4c83c49cc2241bcce066cb7b75967b2"
export WANDB_MODE="offline"
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_DEBUG=TRACE
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="^docker0,lo,bond0"
export MASTER_PORT=12345
# export WORLD_SIZE=$SLURM_NTASKS
# export LOCAL_RANK=$SLURM_LOCALID
# export RANK=$SLURM_LOCALID
# export NODE_RANK=$SLURM_PROCID
export MASTER_ADDR=$head_node_ip
export WORK_DIR=../
export PYTHONPATH=$WORK_DIR

# echo "WORLD_SIZE=$WORLD_SIZE, LOCAL_RANK=$LOCAL_RANK, NODE_RANK=$NODE_RANK, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
current_time=$(date +%Y%m%d%H%M%S)
# might need the latest CUDA
PROJ_ROOT="./"                      # root directory for saving experiment logs
EXPNAME="test_macvid_t2v_512_debug"        # experiment name 
# EXPNAME="test_macvid_t2v_512_3.5m_$current_time"        # experiment name 
DATADIR="configs/training_data/train_data.yaml"   # dataset directory
CONFIG="configs/train_t2v_512_v1.0.yaml"
CKPT_RESUME="/project/suptest/xchiaa/shared_ckpts/VideoCrafter/Text2Video-512/model.ckpt"
# CKPT_RESUME="checkpoints/model.ckpt"
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
lightning.trainer.num_nodes=8 \
--load_from_checkpoint $CKPT_RESUME 