#!/bin/bash
DATADIR="/project/suptest/xchiaa/debug-yq/MACVideoGen/test_macvid_t2v_1024_0.5m_20240224145612"


while true; do
    wandb sync --include-offline /project/suptest/xchiaa/debug-yq/MACVideoGen/test_macvid_t2v_512_3.5m_20240225234040/wandb/offline-*
    sleep 5m
done
