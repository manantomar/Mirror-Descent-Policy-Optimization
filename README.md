# Mirror-Descent-Policy-Optimization

This repository contains the code for MDPO, a trust-region algorithm based on principles of Mirror Descent. It includes two variants, on-policy MDPO and off-policy MDPO, based on the paper [Mirror Descent Policy Optimization](https://arxiv.org/abs/2005.09814).

This implementation makes use of [Tensorflow](https://github.com/tensorflow/tensorflow) and builds over the code provided by [stable-baselines](https://github.com/hill-a/stable-baselines).

# Getting Started

## Prerequisites

All dependencies are provided in a python virtual-env `requirements.txt` file. Majorly, you would need to install `stable-baselines`, `tensorflow`, and `mujoco_py`.

## Installation

1. Install stable-baselines
~~~
pip install stable-baselines[mpi]==2.7.0
~~~

2. [Download](https://www.roboti.us/index.html) and copy MuJoCo library and license files into a `.mujoco/` directory. We use `mujoco200` for this project.

3. Clone MDPO and copy the `mdpo-on` and `mdpo-off` directories inside [this directory](https://github.com/hill-a/stable-baselines/tree/master/stable_baselines).

4. Activate `virtual-env` using the `requirements.txt` file provided.

~~~
source <virtual env path>/bin/activate
~~~

# Example

Use the `run_mujoco.py` script for training MDPO.

On-policy MDPO
~~~
python3 run_mujoco.py --env=Walker2d-v2 --sgd_steps=10
~~~

Off-policy MDPO
~~~
python3 run_mujoco.py --env=Walker2d-v2 --num_timesteps=1e6 --sgd_steps=1000 --klcoeff=1.0 --lam=0.2 --tsallis_coeff=1.0
~~~

# Reference

~~~
@article{tomar2020mirror,
  title={Mirror Descent Policy Optimization},
  author={Tomar, Manan and Shani, Lior and Efroni, Yonathan and Ghavamzadeh, Mohammad},
  journal={arXiv preprint arXiv:2005.09814},
  year={2020}
}
~~~
