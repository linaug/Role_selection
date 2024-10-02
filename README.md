# Role Selection
Code accompanying the paper
> [Autonomous and Adaptive Role Selection for Multi-robot Collaborative Area Search Based on Deep Reinforcement Learning](https://arxiv.org/abs/2312.01747)\
> Zhu, Lina and Cheng, Jiyu and Zhang, Hao and Cui, Zhichao and Zhang, Wei and Liu, Yuehu\
> (Xi'an Jiaotong University)\
> _arXiv: 2312.01747_.

## Installation
Clone the repository, change directory into its root and run:
```
pip install -e .
```
This will install the package and all requirements. It will also set up the entry points we are referring to later in these instructions.

## Training

The policy training follows this scheme:
```
train_policy coverage -t [total time steps in millions]

```
where `-t` is the total number of time steps at which the experiment is to be terminated (note that this is not per call, but total time steps, so if a policy is trained with `train_policy -t 20` and `-o` is a config option (one of `{self_interested, re_adapt}` as can be found in the `alternative_config` key in each of the config files in `config`).

When running each experiment, Ray will print the trial name to the terminal, which looks something like `MultiPPO_coverage_f4dc4_00000`. By default, Ray will create the directory `~/ray_results/MultiPPO` in which the trial with the given name can be found with its checkpoint. 

## Evaluation
We provide one methods for evaluation:

1) `evaluate_coop`: Evaluate cooperative only performance while disabling self-interested agents with and without communication among cooperative agents.

The evaluation is run as
```
evaluate_coop [checkpoint path] [result path] --trials 100
```
for 100 evaluation runs with different seeds. The resulting file is a Pandas dataframe containing the rewards for all agents at every time step. It can be processed and visualized by running `evaluate_plot [pickled data path]`.

Additionally, a checkpoint can be rolled out and rendered for a randomly generated environment with `evaluate_serve [checkpoint_path] --seed 0`. 

## Citation
If you use any part of this code in your research, please cite our paper:
```
@article{zhu2023autonomous,
  title={Autonomous and Adaptive Role Selection for Multi-robot Collaborative Area Search Based on Deep Reinforcement Learning},
  author={Zhu, Lina and Cheng, Jiyu and Zhang, Hao and Cui, Zhichao and Zhang, Wei and Liu, Yuehu},
  journal={arXiv preprint arXiv:2312.01747},
  year={2023}
}
```
