#!/bin/bash
DATADIR="/home/xchiaa/MACVideoGen/pretrain_macvid_t2v_1024_3m_20240212234515/wandb/offline-run-20240213_102139-pretrain_macvid_t2v_1024_3m_20240212234515"


while true; do
    wandb sync $DATADIR
    sleep 5m
done
