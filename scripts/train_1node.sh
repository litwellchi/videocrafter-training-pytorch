current_time=$(date +%Y%m%d%H%M%S)
# might need the latest CUDA
PROJ_ROOT="./"                      # root directory for saving experiment logs
EXPNAME="test_macvid_t2v_512_v2_debug"        # experiment name 
# EXPNAME="test_macvid_t2v_512_3.5m_$current_time"        # experiment name 
DATADIR="configs/training_data/train_data.yaml"   # dataset directory
CONFIG="configs/train_t2v_512_v2.0.yaml"
CKPT_RESUME='/scratch/vgenfmod/shared_ckpts/VideoCrafter/VideoCrafter2/model.ckpt'
# CONFIG="configs/train_t2v_1024_v1.0.yaml"
# CKPT_RESUME="/scratch/vgenfmod/shared_ckpts/VideoCrafter/Text2Video-1024/model.ckpt"
python train_main.py \
--base $CONFIG \
-t --gpus '0,1,2,3,4,5,6,7', \
--name $EXPNAME \
--logdir $PROJ_ROOT \
--auto_resume True \
--load_from_checkpoint $CKPT_RESUME 