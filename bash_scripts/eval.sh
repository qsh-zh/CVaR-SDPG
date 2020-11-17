#!/bin/bash
cd /Ant
tmux new-session -d -s "test" python after_test.py
