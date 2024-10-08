import argparse
import collections.abc
from importlib.machinery import DEBUG_BYTECODE_SUFFIXES
import json
import yaml
from tkinter import N
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ray
import time
import traceback
import matplotlib.image as mpimg
import cv2 as cv
import matplotlib.animation as animation
import random
import glob
import pdb

from pathlib import Path
from queue import Queue
from queue import PriorityQueue
from collections import deque
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
from ray.util.multiprocessing import Pool

DEBUG = False
RANDOM = False

if not DEBUG:
    from .environments.coverage import CoverageEnv
    from .environments.path_planning import PathPlanningEnv
    from .models.adversarial import RoleModel
    from .trainers.multiagent_ppo import MultiPPOTrainer
    from .trainers.random_heuristic import RandomHeuristicTrainer
    from .trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution
else:
    from environments.coverage import CoverageEnv
    from environments.path_planning import PathPlanningEnv
    from models.adversarial import RoleModel
    from trainers.multiagent_ppo import MultiPPOTrainer
    from trainers.random_heuristic import RandomHeuristicTrainer
    from trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
logger = logging.getLogger(__name__)

RAY_ADDRESS_ENV = "RAY_ADDRESS"

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def run_trial(trainer_class=MultiPPOTrainer, checkpoint_path=None, trial=0, min_interest_fracrtion=0.16,min_coverable_fraction=0.6,cfg_update={}, render=False, save_file=None):
    try:
        t0 = time.time()
        cfg = {'env_config': {}, 'model': {}}  

        if checkpoint_path is not None:
            # We might want to run policies that are not loaded from a checkpoint
            # (e.g. the random policy) and therefore need this to be optional
            with open(Path(checkpoint_path).parent/"params.json") as json_file:
                cfg = json.load(json_file)

        if 'evaluation_config' in cfg:
            # overwrite the environment config with evaluation one if it exists
            cfg = update_dict(cfg, cfg['evaluation_config'])
        
        cfg = update_dict(cfg, cfg_update)
        cfg = update_dict(cfg, {'env_config':  {'min_interesting_area_fraction':min_interest_fracrtion}})
        cfg = update_dict(cfg, {'env_config':  {'min_coverable_area_fraction':min_coverable_fraction}})

        trainer = trainer_class(
            env=cfg['env'],
            logger_creator=lambda config: NoopLogger(config, ""),
            config={
                "framework": "torch",
                "seed": trial,
                "num_workers": 0,
                "env_config": cfg['env_config'],
                "model": cfg['model']
            }
        )
        if checkpoint_path is not None:
            aa = checkpoint_path[-4:-1]
            checkpoint_file = Path(checkpoint_path)/('checkpoint-'+os.path.basename(checkpoint_path).split('_')[-1])
            if not RANDOM:
                trainer.restore(str(checkpoint_file)+str(aa))

        envs = {'coverage': CoverageEnv, 'path_planning': PathPlanningEnv}
        env = envs[cfg['env']](cfg['env_config'])
        env.seed(trial)
        obs = env.reset()

        results = []
        totle_reward = []

        explored_ratio = []
        loudian_ratio = []

        for i in range(cfg['env_config']['max_episode_len']):
            if not RANDOM:
                # pdb.set_trace()
                actions = trainer.compute_action(obs)
                obs, reward, done, info = env.step(actions)
            else:
                actions_random = {'primitive':tuple((random.randrange(0, 5)) for i in range(sum(cfg['env_config']['n_agents']))),
                                  'role':tuple((random.randrange(0, 2)) for i in range(sum(cfg['env_config']['n_agents'])))}
                obs, reward, done, info = env.step(actions_random)

            totle_reward.append(reward)

            if render:              
                aa = env.render()
                aa.savefig(save_file+"{}.png".format(i))
            # for j, reward in enumerate(list(info['rewards'].values())):

            for (j, reward),(j__, reward_coverage),(j___, reward_explore),(j____, covered_ratio_), (j_, reward_role) in zip(
                enumerate(list(info['rewards'].values())), 
                enumerate(list(info['rewards_coverage'].values())),
                enumerate(list(info['rewards_explore'].values())),
                enumerate(list(info['covered_ratio'].values())),
                enumerate(list(info['rewards_role'].values()))):

                results.append({
                    'step': i,
                    'agent': j,
                    'trial': trial,
                    'reward': reward,
                    'reward_role': reward_role,
                    'reward_coverage': reward_coverage,
                    'reward_explore': reward_explore,
                    'covered_ratio': covered_ratio_,
                    'explored_ratio':info['explored_ratio'],
                    'role_cover_ratio':info['roles_cover_ratio'],
                })

            if i == cfg['env_config']['max_episode_len']:
                explored_ratio.append(info['explored_ratio'])
                loudian_ratio.append(info['rewards_loudian'][1][-1])
        
        ##### output vedio ####
        if render:  
            fig = plt.figure()
            ims = []
            for ii in range(cfg['env_config']['max_episode_len']):
                # 用opencv读取图片
                img = cv.imread(save_file+str(ii)+'.png')
                (r, g, b) = cv.split(img) 
                img = cv.merge([b,g,r])  
                im = plt.imshow(img, animated=True)
                ims.append([im])
            # ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=False)
            # ani.save(os.path.join(save_file,'image0/')+"movie0.mp4")
            plt.show()

            # filelist = glob.glob(os.path.join(save_file, "*.png"))
            # for f in filelist:
            #     os.remove(f)

        print("Done_time", time.time() - t0)      
        # print("Average_reward:",sum(totle_reward)/len(totle_reward)) 

    except Exception as e:
        print(e, traceback.format_exc())
        raise
    df = pd.DataFrame(results)
    return df

def path_to_hash(path):
    path_split = path.split('/')
    checkpoint_number_string = path_split[-1].split('_')[-1]
    path_hash = path_split[-2].split('_')[-2]
    return path_hash + '-' + checkpoint_number_string

def serve_config(checkpoint_path, trials, min_interest_fraction,min_coverable_fraction,cfg_change={}, trainer=MultiPPOTrainer):
    with Pool(processes=8) as p:
        results = pd.concat(p.starmap(run_trial, [(trainer, checkpoint_path, t, min_interest_fraction,min_coverable_fraction,cfg_change) for t in range(trials)]))
    return results

def serve_config_debug(checkpoint_path, trials, min_interest_fraction,min_coverable_fraction,cfg_change={}, trainer=MultiPPOTrainer):
    reward_all = []
    for t in range(trials):
        results = run_trial(trainer, checkpoint_path, t, min_interest_fraction,min_coverable_fraction,cfg_change)
    return results

def initialize():
    ray.init()
    register_env("coverage", lambda config: CoverageEnv(config))
    register_env("path_planning", lambda config: PathPlanningEnv(config))
    ModelCatalog.register_custom_model("role_model", RoleModel)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)

def eval_nocomm(env_config_func, prefix):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out_file")
    parser.add_argument("--interest","--min_interesting_area_fraction",type=float, default=0.16)
    parser.add_argument("--coverable","--min_coverable_area_fraction",type=float, default=0.6)
    parser.add_argument("-t", "--trials", type=int, default=100)
    args = parser.parse_args()

    # print('*-'*80)
    # print('seed_input:',args.seed)

    initialize()
    results = []
    for comm in [False, True]:
        cfg_change={'env_config': env_config_func(comm)}
        if not DEBUG:
            df = serve_config(args.checkpoint, args.trials, args.interest,args.coverable,cfg_change=cfg_change, trainer=MultiPPOTrainer)
        else:
            df = serve_config_debug(args.checkpoint, args.trials, args.interest,args.coverable,cfg_change=cfg_change, trainer=MultiPPOTrainer)
        df['comm'] = comm
        results.append(df)

    with open(Path(args.checkpoint).parent/"params.json") as json_file:
        cfg = json.load(json_file)
        if 'evaluation_config' in cfg:
            update_dict(cfg, cfg['evaluation_config'])
        update_dict(cfg, {'env_config':  {'min_interesting_area_fraction': args.interest}})
        update_dict(cfg, {'env_config':  {'min_coverable_area_fraction': args.coverable}})

    df = pd.concat(results)
    df.attrs = cfg
    filename = prefix + "-" + "interst" +"_"+str(cfg['env_config']['min_interesting_area_fraction']) + "-" + "coverable" +"_"+str(cfg['env_config']['min_coverable_area_fraction'])+"-"+ "robots" + "_" +str(cfg['env_config']['n_agents'][1]) +"-"+str(cfg['env_config']['ALPHA']) + "explore" + "_" + str(cfg['env_config']['BETA']) + "cover" + "-" + path_to_hash(args.checkpoint) + ".pkl"
    # filename = 'Random' + "-" + "interst" +"_"+str(cfg['env_config']['min_interesting_area_fraction']) + "-" + "coverable" +"_"+str(cfg['env_config']['min_coverable_area_fraction'])+"-"+ "robots" + "_" +str(cfg['env_config']['n_agents'][1]) +"-"+str(cfg['env_config']['ALPHA']) + "explore" + "_" + str(cfg['env_config']['BETA']) + "cover" + "-" + path_to_hash(args.checkpoint) + ".pkl"
    # os.makedirs(os.path.join(args.out_path, "eval_coop-checkpoint-.pkl"), exist_ok = True)           #makedirs 创建文件时如果路径不存在会创建这个路径
    df.to_pickle(Path(args.out_file)/filename)

    # filename_ = prefix + "-" + path_to_hash(args.checkpoint) + ".txt"
    # df.to_csv(Path(args.out_file)/filename_, sep='\t', index=True)

def eval_nocomm_coop():
    # Cooperative agents can communicate or not (without comm interference from adversarial agent)
    eval_nocomm(lambda comm: {
        'disabled_teams_comms': [True, not comm],
        'disabled_teams_step': [True, False]
    }, "eval_coop")

def plot_agent(ax, df, color, step_aggregation='sum', linestyle='-'):
    world_shape = df.attrs['env_config']['world_shape']
    # max_cov = world_shape[0]*world_shape[1]*df.attrs['env_config']['min_coverable_area_fraction']
    max_cov = 0.416 * 25 *25
    d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward'].apply(step_aggregation, 'step').groupby('trial').cumsum()/max_cov*100).groupby('step')
    # df.sort_values是整理给出的栏，如上就是整理trail和step栏.
    ax.plot(d.mean(), color=color, ls=linestyle)
    ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_agent_duplicate(ax, df, color, step_aggregation='sum', linestyle='-'):
    world_shape = df.attrs['env_config']['world_shape']
    # max_cov = world_shape[0]*world_shape[1]*df.attrs['env_config']['min_coverable_area_fraction']
    max_cov = 0.83 * 24 *24
    d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['duplicate_coverage_reward'].apply(step_aggregation, 'step').groupby('trial').cumsum()/max_cov*100).groupby('step')
    # df.sort_values是整理给出的栏，如上就是整理trail和step栏.
    ax.plot(d.mean(), color=color, ls=linestyle)
    ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("data2")
    parser.add_argument("-o", "--out_file", default=None)
    args = parser.parse_args()

    fig_overview = plt.figure(figsize=[4, 4])
    ax = fig_overview.subplots(1, 1)

    df = pd.read_pickle(args.data)
    df2 = pd.read_pickle(args.data2)
    if Path(args.data).name.startswith('eval_adv'):
        plot_agent(ax, df[(df['comm'] == False) & (df['agent'] == 0)], 'r', step_aggregation='mean', linestyle=':')
        plot_agent(ax, df[(df['comm'] == False) & (df['agent'] > 0)], 'b', step_aggregation='mean', linestyle=':')
        plot_agent(ax, df[(df['comm'] == True) & (df['agent'] == 0)], 'r', step_aggregation='mean', linestyle='-')
        plot_agent(ax, df[(df['comm'] == True) & (df['agent'] > 0)], 'b', step_aggregation='mean', linestyle='-')
    elif Path(args.data).name.startswith('eval_coop'):
        plot_agent(ax, df[(df['comm'] == False) & (df['agent'] > 0)], 'b', step_aggregation='sum', linestyle=':')
        plot_agent(ax, df[(df['comm'] == True) & (df['agent'] > 0)], 'b', step_aggregation='sum', linestyle='-')

        plot_agent(ax, df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'r', step_aggregation='sum', linestyle=':')
        plot_agent(ax, df2[(df2['comm'] == True) & (df2['agent'] > 0)], 'r', step_aggregation='sum', linestyle='-')

    elif Path(args.data).name.startswith('eval_rand'):
        plot_agent(ax, df[df['agent'] > 0], 'b', step_aggregation='sum', linestyle='-')

    ax.set_ylabel("Coverage %")
    # ax.set_ylim(0, 300) # 量化重复覆盖
    ax.set_ylim(0, 100)
    ax.set_xlabel("Episode time steps")
    ax.margins(x=0, y=0)
    ax.grid()

    fig_overview.tight_layout()
    if args.out_file is not None:
        fig_overview.savefig(args.out_file, dpi=300)

    plt.show()

def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--interest","--min_interesting_area_fraction",type=float, default=0.24)
    parser.add_argument("--coverable","--min_coverable_area_fraction",type=float, default=0.6)
    parser.add_argument("-o", "--out_file", default=None)
    args = parser.parse_args()
    directory = args.out_file

    idx = 0
    while os.path.exists(f"directory/image%s" % idx):
        idx = idx + 1   

    try:
        os.makedirs(directory, exist_ok = True)
        os.makedirs(os.path.join(directory, "image%s" % idx), exist_ok = True)
    except OSError as error:
        print("Directory '%s' can not be created")

    # save_file = f"directory/image%s" % idx
    # print(save_file)
    initialize()
    run_trial(checkpoint_path=args.checkpoint, trial=args.seed,min_interest_fracrtion=args.interest, min_coverable_fraction=args.coverable,render=True, save_file=directory)

if __name__ == '__main__':
    eval_nocomm_coop() # 无自私机器人评估
    # eval_nocomm_adv() # 有自私机器人
    
    # serve() # for output vedio
    exit()


#     git branch -m main master
# git fetch origin
# git branch -u origin/master master
# git remote set-head origin -a