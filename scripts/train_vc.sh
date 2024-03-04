
PROJ_ROOT="./"                      # root directory for saving experiment logs
# EXPNAME="cv_macvid"          # experiment name 
EXPNAME="cv_webvid"          # experiment name 
# DATADIR="/dataset/sky_timelapse"  # dataset directory
# DATADIR="/aifs4su/mmdata/rawdata/videogen/macvid/video_dataset_85"  # dataset directory
DATADIR="/aifs4su/mmcode/videogen/MACVideoGen/configs/training_data/data_config.yaml" # dataset directory
# AEPATH="models/lvdm_short/short_taihi.ckpt"    # pretrained video autoencoder checkpoint
# AEPATH="models/ae/ae_sky.ckpt"    # pretrained video autoencoder checkpoint

# CONFIG="configs/lvdm_short/sky.yaml"
# OR CONFIG="configs/videoae/ucf.yaml"
CONFIG="configs/train_t2v_1024_v1.0.yaml"
# CKPT_RESUME="/aifs4su/mmcode/videogen/share_ckpts/VideoCrafter/Text2Video-1024/model.ckpt"
CKPT_RESUME="checkpoints/model.ckpt"
# run
export TOKENIZERS_PARALLELISM=false
python train_main.py \
--base $CONFIG \
-t --gpus '0,1,2,3,5,6', \
--name $EXPNAME \
--logdir $PROJ_ROOT \
--auto_resume True \
--wandb True \
lightning.trainer.num_nodes=1 \
--load_from_checkpoint $CKPT_RESUME \ 
# data.params.train.params.data_root $DATADIR \ 
# data.params.validation.params.data_root $DATADIR 
# model.params.first_stage_config.params.ckpt_path=$AEPATH

# -t --gpus '0,1,2,3,4,5,6,7', \
# -------------------------------------------------------------------------------------------------
# commands for multi nodes training
# - use torch.distributed.run to launch main.py
# - set `gpus` and `lightning.trainer.num_nodes`

# For example:

# python -m torch.distributed.run \
#     --nproc_per_node=8 --nnodes=$NHOST --master_addr=$MASTER_ADDR --master_port=1234 --node_rank=$INDEX \
#     main.py \
#     --base $CONFIG \
#     -t --gpus 0,1,2,3,4,5,6,7 \
#     --name $EXPNAME \
#     --logdir $PROJ_ROOT \
#     --auto_resume True \
#     lightning.trainer.num_nodes=$NHOST \
#     data.params.train.params.data_root=$DATADIR \
#     data.params.validation.params.data_root=$DATADIR
