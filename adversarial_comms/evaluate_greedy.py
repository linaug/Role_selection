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
    from .environments.coverage_evaluate import CoverageEnv
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


class traditional_plan():
    def __init__(self):
        self.goal=[-1,-1]
        self.action_quene=[]
        self.grid_quene=[]
        self.neighbors=([0,0],[0,1],[0,-1],[-1,0],[1,0])    #用来表示全局，行动后将会到达哪里。

    def A_star(self,graph):                    #这个是A*算法，根据地图计算路径。
        #先把图转过来吧
        graph_new=[]
        for i in graph:
            i=i.T
            graph_new.append(i)
        graph_new=np.stack(graph_new,axis=1)

        #判断数组是否越界
        def is_in_size(input):
            if input[0]<9 and input[1]<9 and input[0]>-1 and input[1]>-1:
                return True
            else :
                return False

        self.goal=[-1,-1]
        self.last_goal=self.goal
        self.last_action_quene=self.action_quene.copy()
        self.action_quene=[]

        #寻找目标点
        breakflag=0
        frontier_graph=graph_new[2]
        interest_graph=graph_new[1]
        obstacle_graph=graph_new[0]
        #print(interest_graph)
        start=[1,1]
        neighbors=([0,1],[0,-1],[-1,0],[1,0])
        frontier=Queue()
        frontier.put(start)
        reached=[]
        reached.append(start)
        while not frontier.empty():
            current=frontier.get()
            for next in neighbors:
                current_next=[current[0]+next[0],current[1]+next[1]]
                #print(current_next)
                if current_next not in reached and is_in_size(current_next) and obstacle_graph[current_next[0]][current_next[1]]==0:
                    if interest_graph[current_next[0]][current_next[1]]==1 or frontier_graph[current_next[0]][current_next[1]]==1:
                        self.goal=current_next
                        breakflag=1
                    frontier.put(current_next)
                    reached.append(current_next)    
            if breakflag==1:
                break


        #寻找路径
        #print(self.goal)
        start=[1,1]
        frontier_obstacle=PriorityQueue()
        frontier_obstacle.put(start,0)
        came_from=dict()
        cost_so_far=dict()
        came_from[tuple(start)]=None
        cost_so_far[tuple(start)]=0
        next_grid=[1,1]
        while not frontier_obstacle.empty():
            current=frontier_obstacle.get()
            if current ==self.goal:
                break
            for next in neighbors:
                current_next=[current[0]+next[0],current[1]+next[1]]
                if is_in_size(current_next) and obstacle_graph[current_next[0]][current_next[1]]==0:
                    new_cost=cost_so_far[tuple(current)]+1
                    if tuple(current_next) not in cost_so_far or new_cost<cost_so_far[tuple(current_next)]:
                        cost_so_far[tuple(current_next)]=new_cost
                        priority=new_cost+heuristic(self.goal,current_next)
                        frontier_obstacle.put(current_next,priority)
                        came_from[tuple(current_next)]=current

        #计算下一步的动作
        current=self.goal
        #print(self.goal)
        path=[]
        #print(came_from[tuple(current)])
        if not tuple(current) in came_from:
            action=0
            #print("hello")
            self.action_quene.append(action)
            self.grid_quene.append(start)
        else:
            while current !=start:
                path.append(current)
                current=came_from[tuple(current)]

            while len(path)!=0:
                action=0
                next_grid=path[-1]
                path.pop()
                dir=[next_grid[0]-start[0],next_grid[1]-start[1]]
                for index,value in enumerate(neighbors):
                    if value==dir:
                        action=index+1
                        start=next_grid
                        self.action_quene.append(action)
                        self.grid_quene.append(next_grid)


        if len(self.action_quene)<=len(self.last_action_quene) or len(self.last_action_quene)==0:
            action=self.action_quene.pop(0)
            #print("new_goal:")
            #print(self.goal)
        else :
            action=self.last_action_quene.pop(0)
            #print("old_goal")
            #print(self.last_goal)
        #next_grid=self.grid_quene.pop(0)
        #print(self.goal)
        return action

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d               #这个应该修改多层嵌套字典中的值

def traditional_action(obs,plan):        #根据传递过来的状态，用来计算返回的动作
    action=[]
    action.append(random.randrange(0, 5))
    start=[1,1]
    next_pose_list=[[-2,-2],]
    current_pose_list=[[-2,-2],]
    for index,value in enumerate(obs['agents']):
        if not index==0:
            actions=plan[index].A_star(obs['agents'][index]['map'])        #返回next_pose  
            action.append(actions)
            pose_global=obs['agents'][index]['pos']
            #print(pose_global)
            current_pose_list.append(pose_global)
            next_global=[pose_global[0]+plan[index].neighbors[actions][0],pose_global[1]+plan[index].neighbors[actions][1]]
            next_pose_list.append(next_global)
    #检查下一个去到的位置会不会重复
    #print(next_pose_list)
    #print(action)
    for i in [x for x in next_pose_list if next_pose_list.count(x)>1]:
        index=next_pose_list.index(i)
        plan[index].action_quene.insert(0,action[index])
        #print(plan[index].action_quene)
        action[index]=0
        '''                          #这个是为了防止重复，即防止碰撞
        for index2,ii in enumerate(next_pose_list):
            if ii[0]==current_pose_list[index][0] and ii[1]==current_pose_list[index][1]:
                plan[index2].action_quene.insert(0,action[index2])
                action[index2]=0
                for index3,iii in enumerate(next_pose_list):
                    if iii[0]==current_pose_list[index2][0] and iii[1]==current_pose_list[index2][1]:
                        plan[index3].action_quene.insert(0,action[index3])
                        action[index3]=0
        '''
        

    #print(action)
    return tuple(action)

def heuristic(a,b):                  #启发函数，这个是用平方差，来计算的
    return abs(a[0]-b[0])+abs(a[0]-b[0])


#这个函数是主要执行的部分，最后返回一些result
def run_trial(trainer_class=MultiPPOTrainer, checkpoint_path=None, trial=0, cfg_update={}, render=False, save_file=None,traditional=False,traditional_path=None,video_path=None):
    try:
        t0 = time.time()
        cfg = {'env_config': {}, 'model': {}}


        if traditional:
            tratition_plan=[]          #创建五个用来放队列的类
            for i in range(5):
                a=traditional_plan()
                tratition_plan.append(a)

        if traditional_path is not None:                     
            #tratition_plan=traditional_plan()      #创建一个传统规划的类，用来放行动集
            #不能只创建一个类，应该要多创建几个类。应该像每一个，
            with open(traditional_path, "rb") as config_file:
                cfg = yaml.safe_load(config_file)    
        if checkpoint_path is not None:
            # We might want to run policies that are not loaded from a checkpoint
            # (e.g. the random policy) and therefore need this to be optional
            with open(Path(checkpoint_path).parent/"params.json") as json_file:
                cfg = json.load(json_file)
            #作用是读取配置文件
        if 'evaluation_config' in cfg:
            # overwrite the environment config with evaluation one if it exists
            cfg = update_dict(cfg, cfg['evaluation_config'])
        
        cfg = update_dict(cfg, cfg_update)

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
            if not RANDOM or traditional:
                trainer.restore(str(checkpoint_file)+str(aa))

        envs = {'coverage': CoverageEnv, 'path_planning': PathPlanningEnv}
        env = envs[cfg['env']](cfg['env_config'])       #创造环境，对环境进行创造
        env.seed(trial)
        obs = env.reset()

        results = []
        totle_reward = []     #强化学习的部分，这里挺重要的，主要执行部分
        #要实现传统的路径规划的话，应该改这里
        print(cfg['env_config']['max_episode_len'])
        for i in range(cfg['env_config']['max_episode_len']):
            if traditional:
                #print(i)
                actions=traditional_action(obs,tratition_plan)
                #print(actions)
                obs, reward, done, info = env.step(actions)
            elif not RANDOM and not traditional:
                actions = trainer.compute_action(obs)
                print(actions)
                obs, reward, done, info = env.step(actions)
            else:
                actions_random = tuple((random.randrange(0, 5)) for i in range(sum(cfg['env_config']['n_agents']))) #action为（五个行动序列的元组）
                obs, reward, done, info = env.step(actions_random)
            totle_reward.append(reward)
            # results.append({
            #     'step':i,
            #     'trial': trial,
            #     'reward': reward
            # })
            # print("The reward sum is {} in timestep {}".format(reward, i))

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
            ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=False)
            ani.save(video_path+'/'+"movie0.mp4")
            plt.show()

            filelist = glob.glob(os.path.join(save_file, "*.png"))
            for f in filelist:
               os.remove(f)

        print("Done_time", time.time() - t0)      
        # print("Average_reward:",sum(totle_reward)/len(totle_reward)) 

    except Exception as e:
        print(e, traceback.format_exc())
        raise
    df = pd.DataFrame(results)
    #print(df)
    return df

def path_to_hash(path):
    path_split = path.split('/')
    checkpoint_number_string = path_split[-1].split('_')[-1]
    path_hash = path_split[-2].split('_')[-2]
    return path_hash + '-' + checkpoint_number_string

def serve_config(checkpoint_path, trials, cfg_change={}, trainer=MultiPPOTrainer,render=False, save_file=None,traditional=False):
    with Pool(processes=8) as p:
        results = pd.concat(p.starmap(run_trial, [(trainer, checkpoint_path, t, cfg_change,render,save_file,traditional) for t in range(trials)]))  
        #在这个函数中，(trainer, checkpoint_path, t, cfg_change)这些是run_trial这个函数的输入，这里需会输出多个量然后通过concat
        #拼在一起，得到最后的结果。
    return results

def serve_config_debug(checkpoint_path, trials, cfg_change={}, trainer=MultiPPOTrainer):
    reward_all = []
    for t in range(trials):
        results = run_trial(trainer, checkpoint_path, t, cfg_change)
    return results

def initialize():
    ray.init()
    register_env("coverage", lambda config: CoverageEnv(config))
    register_env("path_planning", lambda config: PathPlanningEnv(config))
    ModelCatalog.register_custom_model("role_model", RoleModel)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)       #初始化，还是环境，模型和动作

def eval_nocomm(env_config_func, prefix):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out_path")
    parser.add_argument("--tradition",action='store_true',default=False)
    parser.add_argument("-t", "--trials", type=int, default=100)
    args = parser.parse_args()

    # print('*-'*80)
    # print('seed_input:',args.seed)

    initialize()
    results = []                 #创了一个初始化的环境
    for comm in [False, True]:
        cfg_change={'env_config': env_config_func(comm)}
        if not DEBUG:
            df = serve_config(args.checkpoint, args.trials, cfg_change=cfg_change, trainer=MultiPPOTrainer,traditional=args.tradition)
        else:
            df = serve_config_debug(args.checkpoint, args.trials, cfg_change=cfg_change, trainer=MultiPPOTrainer)
        df['comm'] = comm
        results.append(df)
        #print(results)
    with open(Path(args.checkpoint).parent/"params.json") as json_file:
        cfg = json.load(json_file)           #加载这个配置文件
        if 'evaluation_config' in cfg:
            update_dict(cfg, cfg['evaluation_config'])

    df = pd.concat(results)
    df_1 = df
    
    ########## save average values in txt
    len = cfg['env_config']['max_episode_len']*int(args.trials)*int(sum(cfg['env_config']['n_agents']))  # 345*10
    reward_avg = None
    reward_avg_ = None
    for comm_ in range(2):
        reward_avg = df_1.iloc[int(comm_*len):int((comm_+1)*len)]['reward']
        # np.savetxt('/home/zln/adv_results/coop_0301_02/test_file.txt', reward_avg, fmt = "%d")
        reward_avg_ = reward_avg.to_numpy().astype(np.float64)
        reward_avg_ = np.reshape(reward_avg_, (args.trials, cfg['env_config']['max_episode_len'],sum(cfg['env_config']['n_agents'])))      
        reward_avg_ = np.sum(reward_avg_,axis=2)   
        reward_avg_ = np.sum(reward_avg_,axis=1)       
        reward_avg_ = np.mean(reward_avg_, axis=0)
        print("average_reward:",reward_avg_)
        # np.savetxt(Path(args.out_path)/"{}.txt".format(comm_), reward_avg_, fmt = "%.18f")

    #创建一个文件夹，用来放数据
    idx = 1
    directory=args.out_path
    exis=os.path.join(directory, "data%s" % idx)
    while  os.path.exists(exis):
        idx = idx + 1   
        exis=os.path.join(directory, "data%s" % idx)
    try:
        exis=os.path.join(directory, "data%s" % idx)
        os.makedirs(directory, exist_ok = True)
        os.makedirs(exis, exist_ok = True)
        #print("ok")
    except OSError as error:
        print("Directory '%s' can not be created")

    df.attrs = cfg
    filename = "Greedy" + "-" + "interst" +"_"+str(cfg['env_config']['min_interesting_area_fraction']) + "-" + "coverable" +"_"+str(cfg['env_config']['min_coverable_area_fraction'])+"-"+ "robots" + "_" +str(cfg['env_config']['n_agents'][1]) +"-"+str(cfg['env_config']['ALPHA']) + "explore" + "_" + str(cfg['env_config']['BETA']) + "cover" + "-" + path_to_hash(args.checkpoint) + ".pkl"
    #os.makedirs(os.path.join(args.out_path, "eval_coop-checkpoint-.pkl"), exist_ok = True)           #makedirs 创建文件时如果路径不存在会创建这个路径
    df.to_pickle(Path(exis)/filename)

    # filename_ = prefix + "-" + path_to_hash(args.checkpoint) + ".csv"
    # df.to_csv(Path(exis)/filename_, sep='\t', index=True)

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
    #d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward'].apply(step_aggregation, 'step').groupby('trial').cumsum()/max_cov*100).groupby('step')
    #d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward'].apply(step_aggregation, 'step').groupby('trial').cumsum()).groupby('step')
    #print(df['reward'])
    #print(d.mean())
    # df.sort_values是整理给出的栏，如上就是整理trail和step栏.
    
    #打印数据,平均值
    d=df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward'].apply('sum', 'step').groupby('trial').sum()
    print(d.mean())
    ax.plot(d.mean(), color=color, ls=linestyle)
    #ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_agent_duplicate(ax, df, color, step_aggregation='sum', linestyle='-'):
    world_shape = df.attrs['env_config']['world_shape']
    # max_cov = world_shape[0]*world_shape[1]*df.attrs['env_config']['min_coverable_area_fraction']
    max_cov = 0.83 * 24 *24
    #这个是覆盖率，即所到的兴趣点占所有点的比例
    #d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['duplicate_coverage_reward'].apply(step_aggregation, 'step').groupby('trial').cumsum()/max_cov*100).groupby('step')
    #这个是奖励，所要到达的点
    d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['duplicate_coverage_reward'].apply(step_aggregation, 'step').groupby('trial').cumsum()).groupby('step')
    # df.sort_values是整理给出的栏，如上就是整理trail和step栏.
    ax.plot(d.mean(), color=color, ls=linestyle)
    ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("data2")
    # parser.add_argument("data3")
    parser.add_argument("-o", "--out_file", default=None)
    args = parser.parse_args()

    fig_overview = plt.figure(figsize=[4, 4])
    ax = fig_overview.subplots(1, 1)

    df = pd.read_pickle(args.data)
    df2 = pd.read_pickle(args.data2)
    # df3 = pd.read_pickle(args.data3)
    if Path(args.data).name.startswith('eval_adv'):
        plot_agent(ax, df[(df['comm'] == False) & (df['agent'] == 0)], 'r', step_aggregation='mean', linestyle=':')
        plot_agent(ax, df[(df['comm'] == False) & (df['agent'] > 0)], 'b', step_aggregation='mean', linestyle=':')
        plot_agent(ax, df[(df['comm'] == True) & (df['agent'] == 0)], 'r', step_aggregation='mean', linestyle='-')
        plot_agent(ax, df[(df['comm'] == True) & (df['agent'] > 0)], 'b', step_aggregation='mean', linestyle='-')
    elif Path(args.data).name.startswith('eval_coop'):
        plot_agent(ax, df[(df['comm'] == False) & (df['agent'] > 0)], 'b', step_aggregation='sum', linestyle=':')
        plot_agent(ax, df[(df['comm'] == True) & (df['agent'] > 0)], 'b', step_aggregation='sum', linestyle='-')
        # plot_agent_duplicate(ax, df[(df['comm'] == False) & (df['agent'] > 0)], 'b', step_aggregation='sum', linestyle=':')
        # plot_agent_duplicate(ax, df[(df['comm'] == True) & (df['agent'] > 0)], 'b', step_aggregation='sum', linestyle='-')
        plot_agent(ax, df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'r', step_aggregation='sum', linestyle=':')
        plot_agent(ax, df2[(df2['comm'] == True) & (df2['agent'] > 0)], 'r', step_aggregation='sum', linestyle='-')

        # plot_agent_duplicate(ax, df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'k', step_aggregation='sum', linestyle=':')
        # plot_agent_duplicate(ax, df2[(df2['comm'] == True) & (df2['agent'] > 0)], 'k', step_aggregation='sum', linestyle='-')

    elif Path(args.data).name.startswith('eval_rand'):
        plot_agent(ax, df[df['agent'] > 0], 'b', step_aggregation='sum', linestyle='-')

    ax.set_ylabel("Coverage %") 
    #ax.set_ylim(0, 300) # 量化重复覆盖         
    #输出奖励的情况
    #ax.set_ylabel("reward")
    #ax.set_ylim(0,300)
    #ax.set_ylim(0, 100)
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
    parser.add_argument("-o", "--out_file", default=None)
    args = parser.parse_args()
    directory = args.out_file           
    
    #看一下渲染部分，
    idx = 1
    exis=os.path.join(directory, "image%s" % idx)
    while  os.path.exists(exis):
        idx = idx + 1   
        exis=os.path.join(directory, "image%s" % idx)
    try:
        exis=os.path.join(directory, "image%s" % idx)
        os.makedirs(directory, exist_ok = True)
        os.makedirs(exis, exist_ok = True)
        #print("ok")
    except OSError as error:
        print("Directory '%s' can not be created")
    # save_file = f"directory/image%s" % idx
    # print(save_file)
    print(exis)
    initialize()
    run_trial(checkpoint_path=args.checkpoint, trial=args.seed, render=True, save_file=directory,video_path=exis)

#定义一个传统规划的方法，用来传统算法的思路来解决探索问题
def traditional_evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--experiment",default="/home/zln/repos/adversarial_comms-master/adversarial_comms/config/coverage.yaml")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-r", "--render", action='store_true',default=True)
    parser.add_argument("-o", "--out_file", default="/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/")
    args = parser.parse_args()
    render=args.render
    directory = args.out_file
    experiment_directory=args.experiment 

    if render:
        idx = 1
        exis=os.path.join(directory, "image%s" % idx)
        while  os.path.exists(exis):
            idx = idx + 1   
            exis=os.path.join(directory, "image%s" % idx)
        try:
            exis=os.path.join(directory, "image%s" % idx)
            os.makedirs(directory, exist_ok = True)
            os.makedirs(exis, exist_ok = True)
            print("ok")
        except OSError as error:
            print("Directory '%s' can not be created")
        # save_file = f"directory/image%s" % idx
        # print(save_file)
        print(exis)


    initialize()
    run_trial(trial=args.seed,save_file=directory,traditional=True,traditional_path=experiment_directory,video_path=exis,render=render) 


if __name__ == '__main__':
    #eval_nocomm_coop() # 无自私机器人评估
    # eval_nocomm_adv() # 有自私机器人

    traditional_evaluate()  #传统算法运行
    
    # serve() # for output vedio
    exit()