# CVaR-SDPG

This repo contains code for [paper](https://arxiv.org/abs/2005.00585) (Improving Robustness via Risk Averse Distributional Reinforcement Learning). 

The code is out of maintain. To reproduce the experiment, we recommend to use [Docker Image](https://hub.docker.com/repository/docker/qinsheng/cvar_eval). **We only tested in docker environment.**

## Run

* Ensure you have valid MuJoco key and set it up inside container to evaluate Mujoco tasks.
* `bash_scripts/` contain command to train and evaluate network.

## MICS

* The framework is developed based on [D4PG](https://github.com/msinto93/D4PG)