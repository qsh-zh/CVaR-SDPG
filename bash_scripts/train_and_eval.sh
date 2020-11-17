#!/bin/bash
cd /Ant
tmux new-session -d -s "train" python train.py
sleep 2m
tmux new-session -d -s "test" python test_every_new_ckpt.py
