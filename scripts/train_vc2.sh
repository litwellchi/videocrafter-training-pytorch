nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -p project -t 10:00 --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
module load cuda11.8
export OPENCV_IO_ENABLE_OPENEXR=1
# export NCCL_DEBUG=TRACE
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="^docker0,lo,bond0"
export MASTER_PORT=12398
export MASTER_ADDR=$head_node_ip
export WORK_DIR=./
export LOCAL_WORLD_SIZE=8
export WANDB_API_KEY="231c840bf4c83c49cc2241bcce066cb7b75967b2"
export WANDB_MODE="offline"
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
# export WORLD_SIZE=$SLURM_NTASKS
# export LOCAL_RANK=$SLURM_LOCALID
# export RANK=$SLURM_LOCALID
# export NODE_RANK=$SLURM_PROCID

echo $head_node_ip 
echo $MASTER_ADDR

srun -p project -t 7-00:00:00 -N1 --gpus-per-node 8 --ntasks-per-node 8 --cpus-per-task 4 bash scripts/train_1node.sh
