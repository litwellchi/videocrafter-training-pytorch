#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBTACH --job-name=inference
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH -p project   #important and necessary
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=24:00:00 # must set the training time by default. 24h max...
#SBATCH --cpus-per-task=8
#SBATCH --output=_output/_%j/output.txt
#SBATCH --error=_output/_%j/error.txt
#SBATCH --signal=SIGUSR1@90 # reboot if the process is killed..

# debugging flags (optional)

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
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


name="debug_512_t2v_6k_epoch_00_0228_wed"

# ckpt='/aifs4su/mmcode/videogen/share_ckpts/VideoCrafter/VideoCrafter2/model.ckpt'
ckpt='/project/suptest/xchiaa/debug-yq/MACVideoGen/test_macvid_t2v_512_debug/checkpoints/epoch=0000-step=000018.ckpt'
config='configs/train_t2v_512_v1.0_debug0228.yaml'

# prompt_file="prompts/test_prompts.txt"
prompt_file="prompts/test_prompt.txt"
res_dir="results"

python3 scripts/evaluation/inference.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 4 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28
