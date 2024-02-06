#!/bin/bash
DATADIR="/home/xchiaa/MACVideoGen/test_macvid_t2v_1024_20240207/wandb/offline-run-20240207_071323-test_macvid_t2v_1024_20240207"


while true; do
    wandb sync $DATADIR
    sleep 10m
done