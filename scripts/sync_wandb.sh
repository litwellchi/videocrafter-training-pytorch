#!/bin/bash
DATADIR="/home/xchiaa/MACVideoGen/test_macvid_t2v_1024_3m_20240209011541/wandb/offline-run-20240209_011708-test_macvid_t2v_1024_3m_20240209011541"


while true; do
    wandb sync $DATADIR
    sleep 5m
done